"""Same-state text/visual counterfactuals and one policy-independent T/G/R evaluator."""

import numpy as np
from transformers import set_seed
from .core import ROOT

FAMILIES = {
    "T_native": dict(
        suite="libero_object",
        task="pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        targets=("alphabet_soup_1", "tomato_sauce_1"),
        goals=("basket_1",),
        relation="in",
        texts=(
            "pick up the alphabet soup and place it in the basket",
            "pick up the tomato sauce and place it in the basket",
        ),
    ),
    "G_native": dict(
        suite="libero_goal",
        task="put_the_bowl_on_the_plate",
        targets=("akita_black_bowl_1",),
        goals=("plate_1", "flat_stove_1"),
        relation="on",
        texts=("put the bowl on the plate", "put the bowl on the stove"),
    ),
}
FAMILIES["T_color"] = {
    **FAMILIES["T_native"],
    "targets": ("alphabet_soup_1", "alphabet_soup_2"),
    "texts": (
        "pick up the red can and place it in the basket",
        "pick up the blue can and place it in the basket",
    ),
}
FAMILIES["G_color"] = {
    **FAMILIES["G_native"],
    "goals": ("plate_1", "plate_2"),
    "texts": ("put the bowl on the red plate", "put the bowl on the blue plate"),
}


def bddl_file(family):
    """Reuse LIBERO BDDL and alter only object declarations for the paired color tasks."""
    f = FAMILIES[family]
    source = ROOT / "vendor/libero/libero/libero/bddl_files" / f["suite"] / (f["task"] + ".bddl")
    text = source.read_text()
    if family == "T_color":
        text = text.replace("tomato_sauce_1", "alphabet_soup_2")
        text = text.replace(
            "alphabet_soup_1 - alphabet_soup", "alphabet_soup_1 alphabet_soup_2 - alphabet_soup"
        )
        text = text.replace("    alphabet_soup_2 - tomato_sauce\n", "")
    if family == "G_color":
        text = text.replace("cream_cheese_1", "plate_2")
        text = text.replace("plate_1 - plate", "plate_1 plate_2 - plate")
        text = text.replace("    plate_2 - cream_cheese\n", "")
    path = ROOT / ".cache/bddl" / f"{family}.bddl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def bits(contrast, alternative):
    """Map one text/visual contrast to recipient or donor input bits."""
    return (alternative, 0) if contrast == "text" else (0, alternative)


class Scene:
    """Native LIBERO environment with paired inputs and a condition-independent evaluator."""

    def __init__(self, family, success_hold_steps=5):
        """Build the native environment and configure the stable-placement criterion."""
        from libero.libero.envs import OffScreenRenderEnv

        self.family, self.f = family, FAMILIES[family]
        if success_hold_steps < 1:
            raise ValueError("success_hold_steps must be positive")
        self.success_hold_steps = success_hold_steps
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file(family)), camera_heights=256, camera_widths=256, ignore_done=True
        )
        self.raw = self.env.env

    def reset(self, seed):
        """Reproduce one initial state, stabilize it and reset grasp/placement history."""
        from robosuite.utils.errors import RandomizationError

        for attempt in range(3):
            set_seed(seed)
            self.env.seed(seed)
            try:
                self.obs = self.raw.reset()
                break
            except RandomizationError:
                if attempt == 2:
                    raise
        self.sim = self.raw.sim  # hard reset replaces MjSim!
        self.active_visual = 0
        self.parity = seed % 2 if self.family.endswith("color") else 0
        self.step_number, self.first_grasp, self.grasp_step = 0, None, None
        self.placement_key, self.placement_steps = None, 0
        self.counts = {t: 0 for t in self.f["targets"]}
        self.color_ids = []
        for obj in self.f["targets" if self.family[0] == "T" else "goals"]:
            self.color_ids.append(
                [
                    i
                    for i in range(self.sim.model.ngeom)
                    if (self.sim.model.geom_id2name(i) or "").startswith(obj + "_")
                ]
            )
        self.paint(0)
        for _ in range(10):
            self.obs, _, _, _ = self.env.step(np.array([0.0, 0, 0, 0, 0, 0, -1]))
        self.initial_z = {t: self.position(t)[2] for t in self.f["targets"]}
        self.collision_guard()
        return self.snapshot()

    def position(self, obj):
        """Read an object body position directly from MuJoCo."""
        return self.sim.data.body_xpos[self.raw.obj_body_id[obj]].copy()

    def restore(self, snapshot):
        """Restore a recorded physical state for activation extraction, without stepping physics.

        Call reset with the same scene seed first. Grasp/placement history is not replayed;
        this method is only used for static paired forwards, never to resume a rollout.
        Derived poses/contacts are recomputed by MuJoCo rather than copied from a trace.
        """
        self.sim.set_state_from_flattened(np.asarray(snapshot["state"]))
        self.sim.forward()
        self.obs = self.raw._get_observations(force_update=True)
        if not np.array_equal(self.sim.get_state().flatten(), snapshot["state"]):
            raise RuntimeError("Recorded simulator state did not restore")

    @staticmethod
    def goal_region(goal):
        """Resolve the native basket/stove placement site for relation checks."""
        return {"basket_1": "basket_1_contain_region", "flat_stove_1": "flat_stove_1_cook_region"}.get(
            goal, goal
        )

    def goal_position(self, goal):
        # The stove's burner site is 15 cm away from its body origin.
        """Read the actual placement region center, rather than a displaced body origin."""
        return self.raw.object_states_dict[self.goal_region(goal)].get_geom_state()["pos"].copy()

    def paint(self, visual):
        """Change visual appearance only; mass, geometry and physical state stay fixed."""
        if not self.family.endswith("color"):
            if visual:
                raise ValueError("Visual exchange is only defined for color families")
            return
        palette = ([0.9, 0.03, 0.03, 1.0], [0.03, 0.1, 0.9, 1.0])
        for i, ids in enumerate(self.color_ids):
            self.sim.model.geom_matid[ids] = -1
            self.sim.model.geom_rgba[ids] = palette[i ^ visual ^ self.parity]

    def observation(self, text_bit=0, visual_bit=0, paraphrase=False):
        """Produce native policy inputs while restoring recipient colors after donor rendering."""
        from robosuite.utils.transform_utils import quat2axisangle

        self.paint(visual_bit)
        try:
            o = self.raw._get_observations(force_update=True)
            video = {
                name: o[camera][::-1, ::-1].copy()[None, None]
                for name, camera in [
                    ("image", "agentview_image"),
                    ("wrist_image", "robot0_eye_in_hand_image"),
                ]
            }
        finally:
            self.paint(self.active_visual)
        xyz = o["robot0_eef_pos"]
        axis = quat2axisangle(o["robot0_eef_quat"].copy())
        state = {
            k: np.array([[[v]]], dtype=np.float32)
            for k, v in zip(("x", "y", "z", "roll", "pitch", "yaw"), [*xyz, *axis])
        }
        state["gripper"] = np.asarray(o["robot0_gripper_qpos"], dtype=np.float32)[None, None]
        text = self.f["texts"][text_bit]
        if paraphrase:
            text = text.replace("pick up", "lift up").replace("put ", "set ", 1)
        return {
            "video": video,
            "state": state,
            "language": {"annotation.human.action.task_description": [[text]]},
        }

    def set_visual(self, visual_bit):
        """Select the recipient color mapping for stepping and video recording."""
        self.active_visual = visual_bit
        self.paint(visual_bit)
        self.obs = self.raw._get_observations(force_update=True)

    def grabbed(self, obj):
        """Use robosuite gripper contacts to detect a held object."""
        return bool(self.raw._check_grasp(self.raw.robots[0].gripper, self.raw.get_object(obj).contact_geoms))

    def step(self, action):
        """Advance physics and update consecutive grasp/placement counters exactly once."""
        self.obs, _, _, _ = self.env.step(action)
        self.step_number += 1
        for obj in self.f["targets"]:
            stable = self.grabbed(obj) and self.position(obj)[2] > self.initial_z[obj] + 0.015
            self.counts[obj] = self.counts[obj] + 1 if stable else 0
            if self.first_grasp is None and self.counts[obj] >= 3:
                self.first_grasp, self.grasp_step = obj, self.step_number
        g, predicate, contact, released = self.placement()
        key = (self.first_grasp, g) if predicate and contact and released else None
        self.placement_steps = (self.placement_steps + 1 if key == self.placement_key else 1) if key else 0
        self.placement_key = key

    @property
    def eligible(self):
        """Allow T edits before first stable grasp and G edits after the required grasp."""
        if self.family[0] == "T":
            return self.first_grasp is None
        return self.first_grasp == self.f["targets"][0]

    def desired(self, text_bit, visual_bit):
        """Return the physical target index implied by text, color exchange and scene parity."""
        return text_bit ^ visual_bit ^ self.parity

    def placement(self):
        """Check nearest destination, native relation, actual contact and release."""
        from libero.libero.envs.predicates import eval_predicate_fn

        t, g, relation, contact, released = self.first_grasp, None, False, False, False
        if t is not None:
            distances = [
                np.linalg.norm(self.position(t)[:2] - self.goal_position(v)[:2]) for v in self.f["goals"]
            ]
            nearest = int(np.argmin(distances))
            if distances[nearest] < 0.09:
                g = nearest
                goal = self.f["goals"][g]
                predicate_goal = self.goal_region(goal)
                relation = bool(
                    eval_predicate_fn(
                        self.f["relation"],
                        self.raw.object_states_dict[t],
                        self.raw.object_states_dict[predicate_goal],
                    )
                )
                # A site has no dynamics: LIBERO's site.check_contact always returns True.
                contact = bool(self.raw.check_contact(self.raw.get_object(t), self.raw.get_object(goal)))
                released = not self.grabbed(t)
        return g, relation, contact, released

    def outcome(self, text_bit=0, visual_bit=0):
        """Evaluate T/G/R, semantic alternatives and current-instruction success."""
        t = self.first_grasp
        g, predicate, contact, released = self.placement()
        instant = predicate and contact and released
        relation = (
            instant and self.placement_key == (t, g) and self.placement_steps >= self.success_hold_steps
        )
        desired = self.desired(text_bit, visual_bit)
        ti = self.f["targets"].index(t) if t is not None else None
        choice = ti if self.family[0] == "T" else g
        preserve = (g == 0 if self.family[0] == "T" else ti == 0) and relation
        tc = ti == (desired if self.family[0] == "T" else 0)
        gc = g == (desired if self.family[0] == "G" else 0)
        return dict(
            T=t,
            G=g,
            R=bool(relation),
            R_native=bool(predicate),
            R_instant=bool(instant),
            goal_contact=bool(contact),
            released=bool(released),
            placement_steps=self.placement_steps,
            bddl_success=bool(self.raw._check_success()),
            T_correct=bool(tc),
            G_correct=bool(gc),
            success=bool(tc and gc and relation),
            choice=choice,
            choice0=choice == self.parity,
            choice1=choice == (1 ^ self.parity),
            select0=bool(choice == self.parity and preserve),
            select1=bool(choice == (1 ^ self.parity) and preserve),
            preserve=bool(preserve),
            grasp_step=self.grasp_step,
        )

    def snapshot(self):
        """Record simulator state, object/goal coordinates, gripper position and contacts."""
        return {
            "state": self.sim.get_state().flatten().tolist(),
            "poses": {
                k: [*self.position(k).tolist(), *self.sim.data.body_xquat[v].tolist()]
                for k, v in self.raw.obj_body_id.items()
            },
            "eef": self.obs["robot0_eef_pos"].tolist(),
            "goal_positions": {goal: self.goal_position(goal).tolist() for goal in self.f["goals"]},
            "contacts": [
                [int(c.geom1), int(c.geom2), float(c.dist)]
                for c in self.sim.data.contact[: self.sim.data.ncon]
            ],
        }

    def collision_guard(self):
        """Reject strongly interpenetrating alternatives instead of changing the sampled seed."""
        relevant = {i for ids in self.color_ids for i in ids}
        for c in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                c.geom1 in relevant
                and c.geom2 in relevant
                and c.dist < -0.005
                and self.sim.model.geom_bodyid[c.geom1] != self.sim.model.geom_bodyid[c.geom2]
            ):
                raise ValueError("Invalid scene: relevant objects penetrate by more than 5 mm")

    def frame(self):
        """Return the native camera frame with the same orientation as policy inputs."""
        return self.obs["agentview_image"][::-1, ::-1].copy()

    def close(self):
        """Release LIBERO and its MuJoCo render context."""
        self.env.close()
