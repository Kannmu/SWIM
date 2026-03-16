import os
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import serial.tools.list_ports

from experiment_config import (
    DEFAULT_DEMO_SCAN_TEXT,
    DEFAULT_PORT_SCAN_TEXT,
    INTERVAL_GAP_DURATION_S,
    MODULATION_DEFINITIONS,
    STAIRCASE_REVERSALS_TARGET,
    SUPP1_CONDITIONS,
    TRIAL_STIMULUS_DURATION_S,
)
from experiment_logic import ExperimentLogic
from umh_controller import UMHController

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ExperimentApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SWIM Experiment 2 + Supp-1 Controller")
        self.geometry("1280x860")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.controller = UMHController()
        self.logic = ExperimentLogic(self._resolve_data_dir())

        self.is_running_trial = False
        self.start_response_time = None
        self.current_trial_payload = None

        self.mode_var = tk.StringVar(value=self.logic.get_available_modes()[0])
        self.threshold_choice_var = tk.StringVar(value="")
        self.pairwise_choice_var = tk.StringVar(value="")

        self.create_widgets()

    def _resolve_data_dir(self):
        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "Data")
        os.makedirs(data_dir, exist_ok=True)
        return data_dir

    def create_widgets(self):
        self.tabview = ctk.CTkTabview(self, width=1160, height=800)
        self.tabview.grid(row=0, column=0, padx=16, pady=16, sticky="nsew")

        self.tab_connect = self.tabview.add("Connection & Setup")
        self.tab_experiment = self.tabview.add("Experiment")

        self.tab_connect.grid_columnconfigure(0, weight=1)
        self.tab_experiment.grid_columnconfigure(0, weight=0)
        self.tab_experiment.grid_columnconfigure(1, weight=1)
        self.tab_experiment.grid_rowconfigure(0, weight=1)

        self.setup_connect_tab()
        self.setup_experiment_tab()
        self.refresh_mode_description()
        self.update_ui_state()

    def setup_connect_tab(self):
        container = ctk.CTkFrame(self.tab_connect, fg_color="transparent")
        container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        container.grid_columnconfigure(0, weight=1)

        device_card = ctk.CTkFrame(container, corner_radius=14)
        device_card.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        device_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(device_card, text="Device Connection", font=ctk.CTkFont(size=22, weight="bold")).grid(
            row=0, column=0, columnspan=3, pady=(18, 16), padx=16, sticky="w"
        )

        self.btn_scan = ctk.CTkButton(device_card, text="↻ Scan Ports", command=self.scan_ports, width=130)
        self.btn_scan.grid(row=1, column=0, padx=(16, 8), pady=8, sticky="w")

        self.combo_ports = ctk.CTkComboBox(device_card, values=[DEFAULT_PORT_SCAN_TEXT], width=320)
        self.combo_ports.grid(row=1, column=1, padx=8, pady=8, sticky="ew")
        self.combo_ports.set(DEFAULT_PORT_SCAN_TEXT)

        self.btn_connect = ctk.CTkButton(device_card, text="Connect Device", command=self.connect_device, width=180)
        self.btn_connect.grid(row=1, column=2, padx=(8, 16), pady=8, sticky="e")

        self.lbl_status = ctk.CTkLabel(
            device_card,
            text="Status: Disconnected",
            text_color="#FF6666",
            font=ctk.CTkFont(size=14),
        )
        self.lbl_status.grid(row=2, column=0, columnspan=3, padx=16, pady=(6, 18), sticky="w")

        mapping_card = ctk.CTkFrame(container, corner_radius=14)
        mapping_card.grid(row=1, column=0, sticky="nsew")
        mapping_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(mapping_card, text="Demo Mapping", font=ctk.CTkFont(size=22, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(18, 8)
        )
        ctk.CTkLabel(
            mapping_card,
            text="DLM_2 / DLM_3 / LM_L / LM_C use on-device demos. ULM Mach variants are sent by stimulation packets.",
            justify="left",
            anchor="w",
        ).grid(row=1, column=0, sticky="w", padx=16, pady=(0, 10))

        self.btn_scan_demos = ctk.CTkButton(mapping_card, text="Scan Demos", command=self.scan_demos, width=140)
        self.btn_scan_demos.grid(row=2, column=0, sticky="w", padx=16, pady=(0, 10))

        self.textbox_demos = ctk.CTkTextbox(mapping_card, height=280, font=ctk.CTkFont(family="Consolas", size=13))
        self.textbox_demos.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 18))
        self.textbox_demos.insert("0.0", DEFAULT_DEMO_SCAN_TEXT + "\n")

    def setup_experiment_tab(self):
        frame_left = ctk.CTkFrame(self.tab_experiment, width=360, corner_radius=12)
        frame_left.grid(row=0, column=0, sticky="nsew", padx=(0, 14), pady=0)
        frame_left.grid_propagate(False)

        ctk.CTkLabel(frame_left, text="Experiment Control", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(22, 18))

        mode_frame = ctk.CTkFrame(frame_left, fg_color="transparent")
        mode_frame.pack(fill="x", padx=20)
        ctk.CTkLabel(mode_frame, text="Experiment Mode", anchor="w").pack(fill="x")
        self.combo_mode = ctk.CTkComboBox(
            mode_frame,
            values=self.logic.get_available_modes(),
            variable=self.mode_var,
            command=lambda _: self.refresh_mode_description(),
            height=36,
        )
        self.combo_mode.pack(fill="x", pady=(6, 8))
        self.lbl_mode_desc = ctk.CTkLabel(mode_frame, text="", justify="left", anchor="w", wraplength=300)
        self.lbl_mode_desc.pack(fill="x", pady=(0, 14))

        participant_frame = ctk.CTkFrame(frame_left, fg_color="transparent")
        participant_frame.pack(fill="x", padx=20)
        ctk.CTkLabel(participant_frame, text="Participant ID", anchor="w").pack(fill="x")
        self.entry_pid = ctk.CTkEntry(participant_frame, placeholder_text="e.g. P01", height=36)
        self.entry_pid.pack(fill="x", pady=(6, 14))

        self.btn_start_exp = ctk.CTkButton(frame_left, text="Start Experiment", command=self.start_experiment_flow, height=40)
        self.btn_start_exp.pack(fill="x", padx=20, pady=(0, 8))

        self.btn_next_block = ctk.CTkButton(
            frame_left,
            text="Start Next Block",
            command=self.next_block_action,
            height=40,
            state="disabled",
            fg_color="#E59400",
            hover_color="#C27D00",
        )
        self.btn_next_block.pack(fill="x", padx=20, pady=(0, 16))

        ctk.CTkFrame(frame_left, height=2, fg_color=("gray70", "gray30")).pack(fill="x", padx=20, pady=8)

        self.lbl_mode_runtime = ctk.CTkLabel(frame_left, text="Mode: -", font=ctk.CTkFont(size=16, weight="bold"), text_color="#4DA3FF")
        self.lbl_mode_runtime.pack(pady=(16, 6))
        self.lbl_block_info = ctk.CTkLabel(frame_left, text="Block/Condition: -", font=ctk.CTkFont(size=15, weight="bold"))
        self.lbl_block_info.pack(pady=4)
        self.lbl_trial_info = ctk.CTkLabel(frame_left, text="Trial: -", font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_trial_info.pack(pady=4)
        self.lbl_progress_hint = ctk.CTkLabel(frame_left, text="Progress: -", justify="left")
        self.lbl_progress_hint.pack(pady=(4, 12))

        ctk.CTkLabel(frame_left, text="Sequence Progress").pack(pady=(6, 6))
        self.progress_bar = ctk.CTkProgressBar(frame_left, height=12, progress_color="#2CC985")
        self.progress_bar.pack(fill="x", padx=20)
        self.progress_bar.set(0)

        self.btn_play_trial = ctk.CTkButton(
            frame_left,
            text="▶ PLAY TRIAL",
            command=self.play_trial_sequence,
            height=60,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="#2CC985",
            hover_color="#229965",
        )
        self.btn_play_trial.pack(fill="x", padx=20, pady=(28, 12))

        summary_card = ctk.CTkFrame(frame_left)
        summary_card.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        ctk.CTkLabel(summary_card, text="Threshold Summary", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=12, pady=(12, 6))
        self.textbox_summary = ctk.CTkTextbox(summary_card, height=220, font=ctk.CTkFont(family="Consolas", size=12))
        self.textbox_summary.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        frame_right = ctk.CTkFrame(self.tab_experiment, corner_radius=12)
        frame_right.grid(row=0, column=1, sticky="nsew", padx=5, pady=0)

        ctk.CTkLabel(frame_right, text="Participant Response", font=ctk.CTkFont(size=26, weight="bold")).pack(pady=(20, 18))

        self.lbl_instruction = ctk.CTkLabel(frame_right, text="Start an experiment to see task instructions.", wraplength=700, justify="left")
        self.lbl_instruction.pack(fill="x", padx=24, pady=(0, 12))

        # Container for question cards to ensure they stay above the button
        self.cards_container = ctk.CTkFrame(frame_right, fg_color="transparent")
        self.cards_container.pack(fill="x")

        self.threshold_card = ctk.CTkFrame(self.cards_container, corner_radius=12)
        self.threshold_card.pack(fill="x", padx=24, pady=(8, 12), ipadx=12, ipady=12)
        ctk.CTkLabel(self.threshold_card, text="Experiment 2: 2-IFC Detection", font=ctk.CTkFont(size=18, weight="bold"), text_color="#4DA3FF").pack(anchor="w", padx=10, pady=(6, 4))
        self.lbl_threshold_prompt = ctk.CTkLabel(self.threshold_card, text="Which interval contained the tactile stimulus?", anchor="w", justify="left")
        self.lbl_threshold_prompt.pack(anchor="w", padx=10, pady=(0, 10))
        threshold_opts = ctk.CTkFrame(self.threshold_card, fg_color="transparent")
        threshold_opts.pack(fill="x", padx=10, pady=(0, 8))
        self.rb_interval_a = ctk.CTkRadioButton(threshold_opts, text="Interval A (First)", variable=self.threshold_choice_var, value="A")
        self.rb_interval_a.pack(side="left", padx=24, pady=10)
        self.rb_interval_b = ctk.CTkRadioButton(threshold_opts, text="Interval B (Second)", variable=self.threshold_choice_var, value="B")
        self.rb_interval_b.pack(side="left", padx=24, pady=10)

        self.pairwise_card = ctk.CTkFrame(self.cards_container, corner_radius=12)
        self.pairwise_card.pack(fill="x", padx=24, pady=(8, 12), ipadx=12, ipady=12)
        ctk.CTkLabel(self.pairwise_card, text="Supplementary Experiment 1: 2-AFC Comparison", font=ctk.CTkFont(size=18, weight="bold"), text_color="#4DA3FF").pack(anchor="w", padx=10, pady=(6, 4))
        self.lbl_pairwise_prompt = ctk.CTkLabel(self.pairwise_card, text="Which stimulus was stronger / clearer?", anchor="w", justify="left")
        self.lbl_pairwise_prompt.pack(anchor="w", padx=10, pady=(0, 10))
        pairwise_opts = ctk.CTkFrame(self.pairwise_card, fg_color="transparent")
        pairwise_opts.pack(fill="x", padx=10, pady=(0, 8))
        self.rb_pair_a = ctk.CTkRadioButton(pairwise_opts, text="Stimulus A (First)", variable=self.pairwise_choice_var, value="A")
        self.rb_pair_a.pack(side="left", padx=24, pady=10)
        self.rb_pair_b = ctk.CTkRadioButton(pairwise_opts, text="Stimulus B (Second)", variable=self.pairwise_choice_var, value="B")
        self.rb_pair_b.pack(side="left", padx=24, pady=10)

        self.btn_record = ctk.CTkButton(frame_right, text="Submit Response", command=self.record_response, height=54, font=ctk.CTkFont(size=17, weight="bold"))
        self.btn_record.pack(fill="x", padx=40, pady=(20, 12))

        self.lbl_status_exp = ctk.CTkLabel(frame_right, text="Ready.", text_color="gray", font=ctk.CTkFont(size=13, slant="italic"))
        self.lbl_status_exp.pack(side="bottom", pady=12)

    def refresh_mode_description(self):
        mode_name = self.mode_var.get()
        if mode_name == "Experiment 2":
            text = (
                "Experiment 2 uses a 3-down-1-up staircase with 2-IFC detection. "
                "Strength steps follow the calibrated intensity transfer function instead of naive linear strength spacing."
            )
        else:
            text = (
                "Supplementary Experiment 1 compares five ULM_L Mach numbers in one GUI. "
                "Use the same software and switch modes with this selector or the experiment toggle."
            )
        self.lbl_mode_desc.configure(text=text)

    def scan_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if ports:
            self.combo_ports.configure(values=ports)
            self.combo_ports.set(ports[0])
        else:
            self.combo_ports.configure(values=["No Ports"])
            self.combo_ports.set("No Ports")

    def connect_device(self):
        port = self.combo_ports.get()
        if not port or port in {"No Ports", DEFAULT_PORT_SCAN_TEXT}:
            messagebox.showwarning("Warning", "Please scan and choose a serial port first.")
            return
        try:
            self.controller.connect(port)
            if self.controller.ping():
                self.lbl_status.configure(text=f"Status: Connected to {port}", text_color="#2CC985")
                messagebox.showinfo("Success", "Device connected and pinged successfully.")
            else:
                self.lbl_status.configure(text="Status: Connected but ping failed", text_color="orange")
                messagebox.showwarning("Warning", "Serial port opened, but ping did not return expected data.")
        except Exception as exc:
            self.lbl_status.configure(text=f"Status: {exc}", text_color="#FF6666")
            messagebox.showerror("Error", str(exc))

    def scan_demos(self):
        if not self.controller.is_connected:
            messagebox.showerror("Error", "Connect the device first.")
            return

        self.textbox_demos.delete("0.0", "end")
        self.textbox_demos.insert("end", "Scanning demos...\n")

        def run_scan():
            try:
                demo_map = self.controller.scan_demo_names(max_demo_count=16)
                found_map = {}
                for idx, name in demo_map.items():
                    self.after(0, lambda i=idx, n=name: self.textbox_demos.insert("end", f"Index {i}: {n}\n"))
                    lower_name = name.lower()
                    for stim_name in MODULATION_DEFINITIONS:
                        if stim_name.lower() in lower_name:
                            found_map[stim_name] = idx
                for stim_name, idx in found_map.items():
                    MODULATION_DEFINITIONS[stim_name]["demo_index"] = idx
                self.after(0, lambda: self.textbox_demos.insert("end", f"\nMapped demos: {found_map}\n"))
            except Exception as exc:
                self.after(0, lambda: self.textbox_demos.insert("end", f"Scan failed: {exc}\n"))

        threading.Thread(target=run_scan, daemon=True).start()

    def start_experiment_flow(self):
        participant_id = self.entry_pid.get().strip()
        if not participant_id:
            messagebox.showwarning("Warning", "Enter Participant ID first.")
            return

        mode_name = self.mode_var.get()
        self.logic.start_experiment(participant_id, mode_name)
        self.btn_start_exp.configure(state="disabled")
        self.btn_next_block.configure(state="disabled")
        self.threshold_choice_var.set("")
        self.pairwise_choice_var.set("")
        self.current_trial_payload = None
        self.update_ui_state()
        messagebox.showinfo("Experiment Started", self.logic.get_mode_label())

    def next_block_action(self):
        if self.logic.next_block():
            self.btn_next_block.configure(state="disabled")
            self.current_trial_payload = None
            self.update_ui_state()
        else:
            messagebox.showinfo("Info", "Experiment complete.")
            self.update_ui_state()

    def update_ui_state(self):
        mode_label = self.logic.get_mode_label()
        self.lbl_mode_runtime.configure(text=f"Mode: {mode_label}")

        current = self.logic.get_current_trial()
        self.current_trial_payload = current

        if self.logic.current_phase == "threshold":
            self.threshold_card.pack(fill="x", padx=24, pady=(8, 12), ipadx=12, ipady=12)
            self.pairwise_card.pack_forget()
            self._update_threshold_ui(current)
        elif self.logic.current_phase == "pairwise":
            self.pairwise_card.pack(fill="x", padx=24, pady=(8, 12), ipadx=12, ipady=12)
            self.threshold_card.pack_forget()
            self._update_pairwise_ui(current)
        else:
            self.threshold_card.pack_forget()
            self.pairwise_card.pack_forget()
            self.lbl_block_info.configure(text="Block/Condition: -")
            self.lbl_trial_info.configure(text="Trial: -")
            self.lbl_progress_hint.configure(text="Progress: -")
            self.textbox_summary.delete("0.0", "end")
            self.textbox_summary.insert("0.0", "No experiment running.\n")

        if current:
            self.btn_play_trial.configure(state="normal", text="▶ PLAY TRIAL")
            self.btn_record.configure(state="disabled")
            self.progress_bar.set(0)
            self.lbl_status_exp.configure(text="Press PLAY to start the trial.")
        else:
            self.btn_play_trial.configure(state="disabled", text="▶ PLAY TRIAL")
            self.btn_record.configure(state="disabled")
            if self.logic.current_phase == "pairwise" and not self.logic.is_experiment_complete():
                self.btn_next_block.configure(state="normal")
                self.lbl_status_exp.configure(text="Current block finished. Start the next block.")
            elif self.logic.is_experiment_complete() and self.logic.mode_name is not None:
                self.btn_start_exp.configure(state="normal")
                self.lbl_status_exp.configure(text="Experiment finished.")

    def _update_threshold_ui(self, current):
        summary_lines = []
        for row in self.logic.get_threshold_summary_rows():
            status = "DONE" if row["complete"] else "RUN"
            summary_lines.append(
                f"{row['condition']:<10} | {status} | trials={row['trials']:<2} | rev={row['reversals']:<2} | current={row['strength']:<6.2f} | est={row['estimate']:.2f}"
            )
        self.textbox_summary.delete("0.0", "end")
        self.textbox_summary.insert("0.0", "\n".join(summary_lines) if summary_lines else "No threshold data yet.\n")

        self.lbl_instruction.configure(
            text=(
                "Experiment 2 uses interleaved 3-down-1-up staircases across all conditions. "
                "Each trial contains two 1.5 s intervals separated by a 1.0 s gap. "
                "One interval contains the tactile stimulus and the other is blank. Report whether the stimulus was in A or B."
            )
        )
        self.lbl_block_info.configure(text=self.logic.get_threshold_progress_text())
        if current:
            self.lbl_trial_info.configure(
                text=(
                    f"Global trial: {current['global_trial_index']} | "
                    f"{current['condition_name']} trial: {current['trial_in_condition']} | "
                    f"Reversals: {current['reversal_count']}/{STAIRCASE_REVERSALS_TARGET}"
                )
            )
            self.lbl_progress_hint.configure(
                text=(
                    f"Active staircases = {current['remaining_active_conditions']} | "
                    f"Current strength = {current['strength']:.2f} | "
                    f"normalized intensity = {current['intensity_level']:.3f}"
                )
            )
            self.lbl_threshold_prompt.configure(
                text=(
                    f"Condition: {current['condition_name']}\n"
                    f"Choose the interval that contained the stimulus."
                )
            )
        else:
            self.lbl_trial_info.configure(text="Threshold run complete")
            self.lbl_progress_hint.configure(text="Progress: all conditions completed.")

    def _update_pairwise_ui(self, current):
        self.textbox_summary.delete("0.0", "end")
        self.textbox_summary.insert(
            "0.0",
            "Supplementary Experiment 1 compares ULM_L Mach conditions:\n" + "\n".join(
                f"- {name}: {SUPP1_CONDITIONS[name]['label']}" for name in self.logic.current_condition_order
            ),
        )

        block_type = self.logic.get_current_block_type()
        total_blocks = len(self.logic.block_order)
        current_block = self.logic.current_block_index + 1 if total_blocks else 0
        self.lbl_instruction.configure(
            text=(
                "Supplementary Experiment 1 uses the Experiment 1 style 2-AFC comparison. "
                "After A and B are presented, choose which one felt stronger or clearer according to the current block."
            )
        )
        self.lbl_block_info.configure(text=f"Block: {current_block}/{total_blocks} ({block_type})")
        if current:
            self.lbl_trial_info.configure(text=f"Trial: {current['trial_in_block']} / {len(self.logic.trials)}")
            prompt = "Which stimulus felt STRONGER?" if block_type == "Intensity" else "Which stimulus felt SMALLER / CLEARER?"
            self.lbl_pairwise_prompt.configure(text=f"{prompt}\nCurrent pair is hidden during playback.")
            self.lbl_progress_hint.configure(text="Stimuli are blind-coded as A and B during presentation.")
        else:
            self.lbl_trial_info.configure(text="Block complete")
            self.lbl_progress_hint.configure(text="Progress: current block completed.")

    def play_trial_sequence(self):
        if self.is_running_trial:
            return
        if not self.current_trial_payload:
            messagebox.showinfo("Info", "No active trial.")
            return
        if not self.controller.is_connected:
            messagebox.showerror("Error", "Connect the UMH device first.")
            return

        self.is_running_trial = True
        self.start_response_time = None
        self.btn_play_trial.configure(state="disabled", text="Running...")
        self.btn_record.configure(state="disabled")
        self.progress_bar.set(0)

        def run_sequence():
            try:
                if self.logic.current_phase == "threshold":
                    self._run_threshold_sequence(self.current_trial_payload)
                else:
                    self._run_pairwise_sequence(self.current_trial_payload)
                self.start_response_time = time.time()
                self.after(0, lambda: self.btn_record.configure(state="normal"))
                self.after(0, lambda: self.lbl_status_exp.configure(text="Sequence finished. Submit the response."))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Playback Error", str(exc)))
                try:
                    self.controller.enable_output(False)
                except Exception:
                    pass
            finally:
                self.is_running_trial = False
                self.after(0, lambda: self.btn_play_trial.configure(state="disabled", text="▶ PLAY TRIAL"))

        threading.Thread(target=run_sequence, daemon=True).start()

    def _play_active_stimulus_interval(self, condition, status_text, progress_value=None):
        self.after(0, lambda: self.lbl_status_exp.configure(text=status_text))
        self.controller.enable_output(False)
        self.controller.play_condition(condition)
        time.sleep(0.05)
        self.controller.enable_output(True)
        time.sleep(TRIAL_STIMULUS_DURATION_S)
        self.controller.enable_output(False)
        if progress_value is not None:
            self.after(0, lambda: self.progress_bar.set(progress_value))

    def _play_blank_interval(self, status_text, progress_value=None):
        self.after(0, lambda: self.lbl_status_exp.configure(text=status_text))
        self.controller.enable_output(False)
        time.sleep(TRIAL_STIMULUS_DURATION_S)
        if progress_value is not None:
            self.after(0, lambda: self.progress_bar.set(progress_value))

    def _run_threshold_sequence(self, trial):
        blank_first = trial["target_interval"] == "B"
        condition = dict(self.logic.get_condition_definition(trial["condition_name"]))
        condition["strength"] = trial["strength"]

        if blank_first:
            self._play_blank_interval("Interval A", progress_value=0.33)
            self.after(0, lambda: self.lbl_status_exp.configure(text="Gap"))
            time.sleep(INTERVAL_GAP_DURATION_S)
            self._play_active_stimulus_interval(condition, "Interval B")
        else:
            self._play_active_stimulus_interval(condition, "Interval A", progress_value=0.33)
            self.after(0, lambda: self.lbl_status_exp.configure(text="Gap"))
            time.sleep(INTERVAL_GAP_DURATION_S)
            self._play_blank_interval("Interval B")
        self.after(0, lambda: self.progress_bar.set(1.0))

    def _run_pairwise_sequence(self, trial):
        condition_a = dict(self.logic.get_condition_definition(trial["stimulus_a"]))
        condition_b = dict(self.logic.get_condition_definition(trial["stimulus_b"]))

        self._play_active_stimulus_interval(condition_a, "Playing stimulus A", progress_value=0.5)
        self.after(0, lambda: self.lbl_status_exp.configure(text="Gap"))
        time.sleep(INTERVAL_GAP_DURATION_S)
        self._play_active_stimulus_interval(condition_b, "Playing stimulus B")
        self.after(0, lambda: self.progress_bar.set(1.0))

    def record_response(self):
        if not self.current_trial_payload:
            return

        reaction_time = 0.0
        if self.start_response_time is not None:
            reaction_time = time.time() - self.start_response_time
        self.start_response_time = None

        try:
            if self.logic.current_phase == "threshold":
                chosen = self.threshold_choice_var.get()
                if not chosen:
                    messagebox.showwarning("Warning", "Choose Interval A or B.")
                    return
                result = self.logic.record_threshold_trial(self.current_trial_payload, chosen, reaction_time)
                self.threshold_choice_var.set("")
                if result["condition_complete"]:
                    self.lbl_status_exp.configure(
                        text=(
                            f"{self.current_trial_payload['condition_name']} complete. "
                            f"Threshold ≈ {result['threshold_estimate']:.2f}."
                        )
                    )
                else:
                    correctness = "correct" if result["is_correct"] else "incorrect"
                    self.lbl_status_exp.configure(
                        text=(
                            f"Response {correctness}. Next strength = {result['next_strength']:.2f}. "
                            f"Reversals = {result['reversal_count']}."
                        )
                    )
                self.current_trial_payload = self.logic.next_trial()
            else:
                chosen = self.pairwise_choice_var.get()
                if not chosen:
                    messagebox.showwarning("Warning", "Choose Stimulus A or B.")
                    return
                self.logic.save_pairwise_trial_data(self.current_trial_payload, chosen, reaction_time)
                self.pairwise_choice_var.set("")
                self.current_trial_payload = self.logic.next_trial()
                self.lbl_status_exp.configure(text="Response saved.")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        self.update_ui_state()


if __name__ == "__main__":
    app = ExperimentApp()
    app.mainloop()
