import os
import gc
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
    EcephysProjectCache,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

output_dir = "/media/maria/notsudata/AllenNeuropixels"

# Downloads all session NWB files.
DOWNLOAD_COMPLETE_DATASET = True

# LFP files are huge. Set this to True only if you really want the full thing.
DOWNLOAD_LFP = True

# Optional: restrict to first N sessions while testing.
# Set to None to download everything.
MAX_SESSIONS = None

# Optional: sleep between sessions to be polite / avoid hammering.
SLEEP_BETWEEN_SESSIONS_SECONDS = 0

# Log file for resumability.
progress_csv = os.path.join(output_dir, "download_progress.csv")
failed_csv = os.path.join(output_dir, "failed_downloads.csv")

manifest_path = os.path.join(output_dir, "manifest.json")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def now_string():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def format_seconds(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def load_existing_progress(progress_path):
    if os.path.exists(progress_path):
        return pd.read_csv(progress_path)
    return pd.DataFrame(
        columns=[
            "session_id",
            "status",
            "download_lfp",
            "started_at",
            "finished_at",
            "elapsed_seconds",
            "error",
        ]
    )


def append_progress(progress_path, row):
    df = pd.DataFrame([row])
    file_exists = os.path.exists(progress_path)
    df.to_csv(progress_path, mode="a", header=not file_exists, index=False)


def append_failure(failed_path, row):
    df = pd.DataFrame([row])
    file_exists = os.path.exists(failed_path)
    df.to_csv(failed_path, mode="a", header=not file_exists, index=False)


def already_done(progress_df, session_id, download_lfp):
    """
    If DOWNLOAD_LFP is False, a session counts as done if session NWB was downloaded.
    If DOWNLOAD_LFP is True, a session counts as done only if LFP was also requested
    and completed in that run.
    """
    if progress_df.empty:
        return False

    session_rows = progress_df[
        (progress_df["session_id"].astype(str) == str(session_id))
        & (progress_df["status"] == "success")
    ]

    if session_rows.empty:
        return False

    if not download_lfp:
        return True

    return bool((session_rows["download_lfp"] == True).any())


def print_banner(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print_banner("Allen Neuropixels downloader")
    print(f"Output directory: {output_dir}")
    print(f"Manifest path:    {manifest_path}")
    print(f"Download sessions: {DOWNLOAD_COMPLETE_DATASET}")
    print(f"Download LFP:      {DOWNLOAD_LFP}")
    print(f"Started at:        {now_string()}")

    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

    print("\nFetching session table...")
    sessions = cache.get_session_table()

    print(f"Total number of sessions in Allen table: {len(sessions)}")
    print("\nSession table preview:")
    print(sessions.head())

    if not DOWNLOAD_COMPLETE_DATASET:
        print("\nDOWNLOAD_COMPLETE_DATASET is False, so only metadata was fetched.")
        return

    session_ids = list(sessions.index)

    if MAX_SESSIONS is not None:
        session_ids = session_ids[:MAX_SESSIONS]

    progress_df = load_existing_progress(progress_csv)

    session_ids_to_download = [
        sid for sid in session_ids
        if not already_done(progress_df, sid, DOWNLOAD_LFP)
    ]

    total_requested = len(session_ids)
    total_remaining_at_start = len(session_ids_to_download)
    total_already_done = total_requested - total_remaining_at_start

    print_banner("Download plan")
    print(f"Requested sessions:       {total_requested}")
    print(f"Already completed:        {total_already_done}")
    print(f"Remaining to download:    {total_remaining_at_start}")
    print(f"Progress CSV:             {progress_csv}")
    print(f"Failures CSV:             {failed_csv}")

    if total_remaining_at_start == 0:
        print("\nEverything requested is already marked as completed. Tiny victory goblin.")
        return

    global_start_time = time.time()
    successes = 0
    failures = 0

    for local_idx, session_id in enumerate(session_ids_to_download, start=1):
        absolute_done_before = total_already_done + local_idx - 1
        remaining_before = total_requested - absolute_done_before

        print_banner(
            f"Session {absolute_done_before + 1}/{total_requested} | "
            f"session_id={session_id} | "
            f"remaining including this one: {remaining_before}"
        )

        started_at = now_string()
        session_start_time = time.time()

        try:
            print(f"[{now_string()}] Downloading/loading session NWB for session_id={session_id}...")

            # This downloads the session NWB if it is not already in the cache.
            session = cache.get_session_data(session_id)

            print(f"[{now_string()}] Session object loaded.")

            if DOWNLOAD_LFP:
                print(f"[{now_string()}] DOWNLOAD_LFP=True, downloading/loading LFP files...")

                # session.probes is a table indexed by probe_id.
                probe_ids = list(session.probes.index)
                print(f"Found {len(probe_ids)} probes for session {session_id}: {probe_ids}")

                for probe_idx, probe_id in enumerate(probe_ids, start=1):
                    probes_remaining = len(probe_ids) - probe_idx

                    print(
                        f"[{now_string()}] "
                        f"Session {session_id}: downloading LFP for probe "
                        f"{probe_idx}/{len(probe_ids)} "
                        f"(probe_id={probe_id}, remaining probes={probes_remaining})"
                    )

                    try:
                        lfp = session.get_lfp(probe_id)
                        print(
                            f"[{now_string()}] "
                            f"Finished LFP probe_id={probe_id}. "
                            f"LFP shape/info: {getattr(lfp, 'shape', 'unknown')}"
                        )

                        # Free memory aggressively.
                        del lfp
                        gc.collect()

                    except Exception as probe_error:
                        print(
                            f"[{now_string()}] WARNING: failed LFP download "
                            f"for session_id={session_id}, probe_id={probe_id}"
                        )
                        print(str(probe_error))

                        append_failure(
                            failed_csv,
                            {
                                "session_id": session_id,
                                "probe_id": probe_id,
                                "stage": "lfp",
                                "time": now_string(),
                                "error": repr(probe_error),
                                "traceback": traceback.format_exc(),
                            },
                        )

            elapsed = time.time() - session_start_time
            finished_at = now_string()

            append_progress(
                progress_csv,
                {
                    "session_id": session_id,
                    "status": "success",
                    "download_lfp": DOWNLOAD_LFP,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "elapsed_seconds": round(elapsed, 2),
                    "error": "",
                },
            )

            successes += 1

            # Free memory. EcephysSession can hold large objects.
            del session
            gc.collect()

            total_elapsed = time.time() - global_start_time
            completed_this_run = successes + failures
            remaining_after = total_remaining_at_start - completed_this_run

            avg_per_attempt = total_elapsed / completed_this_run
            estimated_remaining_seconds = avg_per_attempt * remaining_after

            print(
                f"\n[{now_string()}] SUCCESS session_id={session_id} "
                f"in {format_seconds(elapsed)}"
            )
            print(
                f"Completed this run: {completed_this_run}/{total_remaining_at_start} | "
                f"Successes: {successes} | "
                f"Failures: {failures} | "
                f"Remaining this run: {remaining_after}"
            )
            print(
                f"Elapsed this run: {format_seconds(total_elapsed)} | "
                f"Rough ETA for remaining: {format_seconds(estimated_remaining_seconds)}"
            )

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received. Stopping cleanly.")
            print("Progress so far has been saved.")
            raise

        except Exception as error:
            elapsed = time.time() - session_start_time
            finished_at = now_string()

            print(f"\n[{now_string()}] FAILED session_id={session_id}")
            print(f"Error: {repr(error)}")
            print(traceback.format_exc())

            append_progress(
                progress_csv,
                {
                    "session_id": session_id,
                    "status": "failed",
                    "download_lfp": DOWNLOAD_LFP,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "elapsed_seconds": round(elapsed, 2),
                    "error": repr(error),
                },
            )

            append_failure(
                failed_csv,
                {
                    "session_id": session_id,
                    "probe_id": "",
                    "stage": "session",
                    "time": now_string(),
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )

            failures += 1
            gc.collect()

            completed_this_run = successes + failures
            remaining_after = total_remaining_at_start - completed_this_run

            print(
                f"Completed this run: {completed_this_run}/{total_remaining_at_start} | "
                f"Successes: {successes} | "
                f"Failures: {failures} | "
                f"Remaining this run: {remaining_after}"
            )

        if SLEEP_BETWEEN_SESSIONS_SECONDS > 0:
            print(f"Sleeping {SLEEP_BETWEEN_SESSIONS_SECONDS}s before next session...")
            time.sleep(SLEEP_BETWEEN_SESSIONS_SECONDS)

    total_elapsed = time.time() - global_start_time

    print_banner("Download run finished")
    print(f"Finished at:       {now_string()}")
    print(f"Successes:         {successes}")
    print(f"Failures:          {failures}")
    print(f"Elapsed:           {format_seconds(total_elapsed)}")
    print(f"Progress CSV:      {progress_csv}")
    print(f"Failures CSV:      {failed_csv}")


if __name__ == "__main__":
    main()