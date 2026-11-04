#!/usr/bin/env python3
"""
organize_oulu.py
Copies Oulu-CASIA frames from the raw download structure into the
pipeline-standard structure expected by preprocessing_exp_faciale.py.

Source:
    /sandbox1/aramadan/Oulu_Dataset/original_images/VL/strong/{subject}/{emotion}/FirstF_*.jpeg

Destination:
    /sandbox1/aramadan/frames/Oulu/{subject}/{emotion}/FirstF_*.jpeg

Run once, then throw away.
Usage:
    python3 organize_oulu.py
"""

import os
import shutil
from tqdm import tqdm

SRC_ROOT = '/sandbox1/aramadan/preprocessed_images/VL_Acropped/Strong'
DST_ROOT = '/sandbox1/aramadan/frames/Oulu_Acropped'

# Paper: 6 basic emotions, exact folder names as they appear in the dataset
VALID_EMOTIONS = {'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise'}


def main():
    if not os.path.exists(SRC_ROOT):
        raise FileNotFoundError(f'Source not found: {SRC_ROOT}')

    subjects = sorted(os.listdir(SRC_ROOT))
    print(f'Found {len(subjects)} subjects: {subjects[:5]} ...')

    total_copied = 0
    total_skipped = 0
    problems = []

    for subject in tqdm(subjects, desc='Subjects'):
        subject_src = os.path.join(SRC_ROOT, subject)
        if not os.path.isdir(subject_src):
            continue

        emotions = sorted(os.listdir(subject_src))

        for emotion in emotions:
            # Skip anything that is not one of the 6 target emotions
            if emotion not in VALID_EMOTIONS:
                print(f'  [skip] unexpected folder: {subject}/{emotion}')
                continue

            emotion_src = os.path.join(subject_src, emotion)
            emotion_dst = os.path.join(DST_ROOT, subject, emotion)
            os.makedirs(emotion_dst, exist_ok=True)

            frames = sorted(f for f in os.listdir(emotion_src)
                            if f.lower().endswith(('.jpeg', '.jpg', '.png')))

            if len(frames) == 0:
                problems.append(f'{subject}/{emotion} — no frames found')
                continue

            for fname in frames:
                src_path = os.path.join(emotion_src, fname)
                dst_path = os.path.join(emotion_dst, fname)

                if os.path.exists(dst_path):
                    total_skipped += 1
                    continue

                shutil.copy2(src_path, dst_path)
                total_copied += 1

    print(f'\nDone.')
    print(f'  Copied  : {total_copied} frames')
    print(f'  Skipped : {total_skipped} frames (already existed)')

    if problems:
        print(f'\nProblems ({len(problems)}):')
        for p in problems:
            print(f'  {p}')

    # Quick sanity check
    print('\n--- Sanity check ---')
    subjects_dst = sorted(os.listdir(DST_ROOT))
    print(f'Subjects in destination : {len(subjects_dst)}')
    # Show frame count for first subject/emotion pair
    first_subj = subjects_dst[0]
    first_emot = sorted(os.listdir(os.path.join(DST_ROOT, first_subj)))[0]
    first_frames = os.listdir(os.path.join(DST_ROOT, first_subj, first_emot))
    print(f'Example: {first_subj}/{first_emot} → {len(first_frames)} frames')


if __name__ == '__main__':
    main()
