import csv
import datetime
import itertools
import os
import random

from experiment_config import (
    DEFAULT_STEP_SIZE,
    EXPERIMENT2_STRENGTH_GRID,
    EXPERIMENT_MODE_DEFINITIONS,
    INITIAL_STRENGTH,
    MAX_STRENGTH,
    MIN_STRENGTH,
    MODULATION_DEFINITIONS,
    STAIRCASE_CONSECUTIVE_CORRECT_FOR_DOWN,
    STAIRCASE_FINE_STEP_COUNT,
    STAIRCASE_INITIAL_STEP_COUNT,
    STAIRCASE_MAX_TRIALS,
    STAIRCASE_MIN_TRIALS,
    STAIRCASE_REVERSAL_MEAN_COUNT,
    STAIRCASE_REVERSALS_TARGET,
    STAIRCASE_STEP_REDUCTION_REVERSAL,
    SUPP1_CONDITIONS,
    strength_to_intensity,
)


MODE_DATA_SUBDIRS = {
    "Experiment 2": "Experiment_2",
    "Supplementary Experiment 1": "Supplementary_Experiment_1",
}

MODE_FILE_TAGS = {
    "Experiment 2": "experiment2",
    "Supplementary Experiment 1": "supplementary_experiment1",
}


class StaircaseState:
    def __init__(self, condition_name, step_size=DEFAULT_STEP_SIZE):
        self.condition_name = condition_name
        self.step_size = float(step_size)
        self.grid = list(EXPERIMENT2_STRENGTH_GRID)
        self.current_strength = self._nearest_grid_value(INITIAL_STRENGTH)
        self.consecutive_correct = 0
        self.last_direction = None
        self.reversals = []
        self.trial_count = 0
        self.is_complete = False

    def _nearest_grid_value(self, strength):
        return min(self.grid, key=lambda value: abs(value - strength))

    def _get_step_count(self):
        if len(self.reversals) >= STAIRCASE_STEP_REDUCTION_REVERSAL:
            return STAIRCASE_FINE_STEP_COUNT
        return STAIRCASE_INITIAL_STEP_COUNT

    def get_intensity_level(self):
        return strength_to_intensity(self.current_strength)

    def apply_response(self, is_correct):
        if self.is_complete:
            return

        self.trial_count += 1
        move_direction = None

        if is_correct:
            self.consecutive_correct += 1
            if self.consecutive_correct >= STAIRCASE_CONSECUTIVE_CORRECT_FOR_DOWN:
                move_direction = -1
                self.consecutive_correct = 0
        else:
            self.consecutive_correct = 0
            move_direction = 1

        if move_direction is None:
            self._check_completion()
            return

        current_index = self.grid.index(self.current_strength)
        step_count = self._get_step_count()
        new_index = min(max(current_index + (move_direction * step_count), 0), len(self.grid) - 1)
        actual_direction = 0
        if new_index > current_index:
            actual_direction = 1
        elif new_index < current_index:
            actual_direction = -1

        if actual_direction != 0:
            if self.last_direction is not None and actual_direction != self.last_direction:
                self.reversals.append(self.current_strength)
            self.last_direction = actual_direction
            self.current_strength = self.grid[new_index]

        self._check_completion()

    def _check_completion(self):
        enough_reversals = len(self.reversals) >= STAIRCASE_REVERSALS_TARGET
        enough_trials = self.trial_count >= STAIRCASE_MIN_TRIALS
        too_many_trials = self.trial_count >= STAIRCASE_MAX_TRIALS
        self.is_complete = too_many_trials or (enough_reversals and enough_trials)

    def get_threshold_estimate(self):
        if not self.reversals:
            return self.current_strength
        tail = self.reversals[-STAIRCASE_REVERSAL_MEAN_COUNT:]
        return sum(tail) / len(tail)


class ExperimentLogic:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.participant_id = "test"
        self.mode_name = None
        self.mode_definition = None
        self.current_phase = None
        self.current_trial_index = 0
        self.trials = []
        self.block_order = []
        self.current_block_index = 0
        self.stimulus_indices = {}
        self.staircases = {}
        self.current_condition_order = []
        self.completed_condition_order = []
        self.active_condition_name = None
        self.threshold_round_robin_order = []
        self.threshold_round_robin_index = 0
        self.global_threshold_trial_index = 0
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        os.makedirs(self.data_dir, exist_ok=True)

    def _get_mode_data_dir(self):
        if self.mode_name is None:
            return self.data_dir
        subdir = MODE_DATA_SUBDIRS.get(self.mode_name, self.mode_name.replace(" ", "_"))
        mode_data_dir = os.path.join(self.data_dir, subdir)
        os.makedirs(mode_data_dir, exist_ok=True)
        return mode_data_dir

    def _get_data_filename(self):
        mode_tag = MODE_FILE_TAGS.get(self.mode_name, "experiment")
        return os.path.join(self._get_mode_data_dir(), f"{self.participant_id}_{mode_tag}.csv")

    def start_experiment(self, participant_id, mode_name):
        self.participant_id = participant_id
        self.mode_name = mode_name
        self.mode_definition = EXPERIMENT_MODE_DEFINITIONS[mode_name]
        self.current_trial_index = 0
        self.current_block_index = 0
        self.completed_condition_order = []
        self.active_condition_name = None
        self.threshold_round_robin_order = []
        self.threshold_round_robin_index = 0
        self.global_threshold_trial_index = 0

        task_type = self.mode_definition["task_type"]
        if task_type == "threshold":
            self.current_phase = "threshold"
            conditions = list(self.mode_definition["conditions"])
            random.shuffle(conditions)
            self.current_condition_order = conditions
            self.threshold_round_robin_order = list(conditions)
            self.staircases = {name: StaircaseState(name) for name in conditions}
            self.trials = []
            self.block_order = []
        else:
            self.current_phase = "pairwise"
            self.staircases = {}
            self.current_condition_order = list(self.mode_definition["conditions"])
            self.block_order = list(self.mode_definition.get("blocks", []))
            random.shuffle(self.block_order)
            self.generate_current_block_trials()

    def get_mode_label(self):
        if not self.mode_definition:
            return "-"
        return self.mode_definition["label"]

    def get_available_modes(self):
        return list(EXPERIMENT_MODE_DEFINITIONS.keys())

    def get_condition_definition(self, name):
        if name in MODULATION_DEFINITIONS:
            return MODULATION_DEFINITIONS[name]
        return SUPP1_CONDITIONS[name]

    def get_current_block_type(self):
        if self.current_phase != "pairwise":
            return None
        if 0 <= self.current_block_index < len(self.block_order):
            return self.block_order[self.current_block_index]
        return None

    def generate_current_block_trials(self):
        full_trials = list(itertools.permutations(self.current_condition_order, 2))
        random.shuffle(full_trials)
        self.trials = full_trials
        self.current_trial_index = 0

    def get_current_trial(self):
        if self.current_phase == "threshold":
            if self.active_condition_name is None:
                self._advance_to_next_threshold_condition()
            if self.active_condition_name is None:
                return None

            staircase = self.staircases[self.active_condition_name]
            target_interval = random.choice(["A", "B"])
            strength = staircase.current_strength
            trial = {
                "condition_name": self.active_condition_name,
                "target_interval": target_interval,
                "strength": strength,
                "intensity_level": staircase.get_intensity_level(),
                "trial_in_condition": staircase.trial_count + 1,
                "global_trial_index": self.global_threshold_trial_index + 1,
                "remaining_active_conditions": self.get_remaining_threshold_condition_count(),
                "reversal_count": len(staircase.reversals),
            }
            return trial

        if self.current_trial_index < len(self.trials):
            stim_a, stim_b = self.trials[self.current_trial_index]
            return {
                "stimulus_a": stim_a,
                "stimulus_b": stim_b,
                "block_type": self.get_current_block_type(),
                "trial_in_block": self.current_trial_index + 1,
            }
        return None

    def _advance_to_next_threshold_condition(self):
        self.active_condition_name = None
        if not self.threshold_round_robin_order:
            return

        condition_count = len(self.threshold_round_robin_order)
        for offset in range(condition_count):
            index = (self.threshold_round_robin_index + offset) % condition_count
            name = self.threshold_round_robin_order[index]
            staircase = self.staircases[name]
            if staircase.is_complete:
                continue
            self.active_condition_name = name
            self.threshold_round_robin_index = (index + 1) % condition_count
            return

    def get_remaining_threshold_condition_count(self):
        return sum(1 for staircase in self.staircases.values() if not staircase.is_complete)

    def record_threshold_trial(self, trial, chosen_interval, reaction_time):
        staircase = self.staircases[trial["condition_name"]]
        is_correct = chosen_interval == trial["target_interval"]
        pre_strength = staircase.current_strength
        pre_reversals = len(staircase.reversals)
        staircase.apply_response(is_correct)
        self.global_threshold_trial_index += 1
        post_strength = staircase.current_strength
        post_reversals = len(staircase.reversals)
        reversal_happened = post_reversals > pre_reversals
        threshold_estimate = staircase.get_threshold_estimate()

        filename = self._get_data_filename()
        file_exists = os.path.exists(filename)
        with open(filename, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "ParticipantID",
                "Timestamp",
                "ExperimentMode",
                "TaskType",
                "GlobalTrial",
                "Condition",
                "TrialInCondition",
                "TargetInterval",
                "ChosenInterval",
                "Correct",
                "StrengthBefore",
                "StrengthAfter",
                "IntensityLevel",
                "ReversalCount",
                "ReversalHappened",
                "ThresholdEstimate",
                "ReactionTime",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "ParticipantID": self.participant_id,
                "Timestamp": datetime.datetime.now().isoformat(),
                "ExperimentMode": self.mode_name,
                "TaskType": "threshold",
                "GlobalTrial": trial["global_trial_index"],
                "Condition": trial["condition_name"],
                "TrialInCondition": trial["trial_in_condition"],
                "TargetInterval": trial["target_interval"],
                "ChosenInterval": chosen_interval,
                "Correct": int(is_correct),
                "StrengthBefore": pre_strength,
                "StrengthAfter": post_strength,
                "IntensityLevel": trial["intensity_level"],
                "ReversalCount": post_reversals,
                "ReversalHappened": int(reversal_happened),
                "ThresholdEstimate": round(threshold_estimate, 4),
                "ReactionTime": reaction_time,
            })

        if staircase.is_complete and trial["condition_name"] not in self.completed_condition_order:
            self.completed_condition_order.append(trial["condition_name"])
        self.active_condition_name = None

        return {
            "is_correct": is_correct,
            "condition_complete": staircase.is_complete,
            "reversal_count": len(staircase.reversals),
            "threshold_estimate": threshold_estimate,
            "next_strength": staircase.current_strength,
        }

    def save_pairwise_trial_data(self, trial, chosen_a_or_b, reaction_time):
        filename = self._get_data_filename()
        file_exists = os.path.exists(filename)
        block_type = self.get_current_block_type()
        chosen_stimulus = trial["stimulus_a"] if chosen_a_or_b == "A" else trial["stimulus_b"]

        with open(filename, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "ParticipantID",
                "Timestamp",
                "ExperimentMode",
                "TaskType",
                "BlockIndex",
                "BlockType",
                "Trial",
                "StimulusA",
                "StimulusB",
                "ChosenOption",
                "ChosenStimulus",
                "ReactionTime",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "ParticipantID": self.participant_id,
                "Timestamp": datetime.datetime.now().isoformat(),
                "ExperimentMode": self.mode_name,
                "TaskType": "pairwise",
                "BlockIndex": self.current_block_index + 1,
                "BlockType": block_type,
                "Trial": self.current_trial_index + 1,
                "StimulusA": trial["stimulus_a"],
                "StimulusB": trial["stimulus_b"],
                "ChosenOption": chosen_a_or_b,
                "ChosenStimulus": chosen_stimulus,
                "ReactionTime": reaction_time,
            })

    def next_trial(self):
        if self.current_phase == "threshold":
            return self.get_current_trial()
        self.current_trial_index += 1
        return self.get_current_trial()

    def next_block(self):
        self.current_block_index += 1
        if self.current_block_index < len(self.block_order):
            self.generate_current_block_trials()
            return True
        return False

    def is_experiment_complete(self):
        if self.current_phase == "threshold":
            return all(staircase.is_complete for staircase in self.staircases.values())
        return self.current_block_index >= len(self.block_order)

    def get_threshold_progress_text(self):
        if self.current_phase != "threshold":
            return ""
        completed = sum(1 for staircase in self.staircases.values() if staircase.is_complete)
        total = len(self.staircases)
        active = self.get_remaining_threshold_condition_count()
        current = self.active_condition_name or "-"
        return f"Interleaved staircase | current: {current} | active: {active} | complete: {completed}/{total}"

    def get_threshold_summary_rows(self):
        rows = []
        for name in self.current_condition_order:
            staircase = self.staircases.get(name)
            if staircase is None:
                continue
            rows.append({
                "condition": name,
                "complete": staircase.is_complete,
                "trials": staircase.trial_count,
                "reversals": len(staircase.reversals),
                "strength": staircase.current_strength,
                "estimate": staircase.get_threshold_estimate(),
            })
        return rows
