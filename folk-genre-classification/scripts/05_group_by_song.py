"""
=========================================================
Group Songs

Groups all audio segments belonging to the same song.

Author : Dhrupad Kashyap
=========================================================
"""

from pathlib import Path

import pandas as pd

from utils.config import (

    METADATA_DIR,

    LINE_SEPARATOR

)


MASTER_DATASET = (

    METADATA_DIR /

    "master_dataset.csv"

)

SONG_GROUPS = (

    METADATA_DIR /

    "song_groups.csv"

)

REPORT_FILE = (

    METADATA_DIR /

    "song_group_report.txt"

)


# ==========================================================
# LOAD MASTER DATASET
# ==========================================================

def load_master_dataset():

    if not MASTER_DATASET.exists():

        raise FileNotFoundError(

            MASTER_DATASET

        )

    return pd.read_csv(

        MASTER_DATASET

    )


# ==========================================================
# VALIDATE DATASET
# ==========================================================

def validate_dataset(data):

    required_columns = [

        "Genre",

        "Original Song ID",

        "Segment ID",

        "Segment Path",

        "Song Title",

        "Artist"

    ]

    missing = []

    for column in required_columns:

        if column not in data.columns:

            missing.append(column)

    if len(missing):

        raise ValueError(

            f"Missing Columns : {missing}"

        )


# ==========================================================
# GROUP SONGS
# ==========================================================

def group_songs(data):

    print()

    print(LINE_SEPARATOR)

    print("Grouping Songs")

    print(LINE_SEPARATOR)

    grouped = (

        data

        .groupby(

            [

                "Genre",

                "Original Song ID",

                "Song Title",

                "Artist"

            ]

        )

        .agg(

            Segment_Count=(

                "Segment ID",

                "count"

            ),

            Segment_List=(

                "Segment Path",

                lambda x: list(x)

            )

        )

        .reset_index()

    )

    print()

    print(

        f"Grouped Songs : {len(grouped)}"

    )

    print()

    return grouped

    # ==========================================================
# SAVE SONG GROUPS
# ==========================================================

def save_song_groups(grouped):

    grouped.to_csv(

        SONG_GROUPS,

        index=False

    )

    print()

    print(LINE_SEPARATOR)

    print("Song groups saved.")

    print(SONG_GROUPS)

    print(LINE_SEPARATOR)


# ==========================================================
# GENERATE REPORT
# ==========================================================

def generate_report(grouped):

    with open(

        REPORT_FILE,

        "w",

        encoding="utf-8"

    ) as report:

        report.write(

            "SONG GROUP REPORT\n"

        )

        report.write(

            "=" * 60 + "\n\n"

        )

        report.write(

            f"Total Songs : {len(grouped)}\n"

        )

        report.write(

            f"Total Genres : {grouped['Genre'].nunique()}\n\n"

        )

        report.write(

            "Genre-wise Song Count\n"

        )

        report.write(

            "-" * 60 + "\n"

        )

        counts = grouped.groupby(

            "Genre"

        ).size()

        for genre, count in counts.items():

            report.write(

                f"{genre:<15}: {count}\n"

            )

    print()

    print("Report generated.")

    print(REPORT_FILE)

    print()


# ==========================================================
# PRINT SUMMARY
# ==========================================================

def print_summary(grouped):

    print()

    print(LINE_SEPARATOR)

    print("SONG GROUP SUMMARY")

    print(LINE_SEPARATOR)

    print(

        f"Total Songs : {len(grouped)}"

    )

    print(

        f"Genres : {grouped['Genre'].nunique()}"

    )

    print()

    print("Songs per Genre")

    print("-" * 60)

    counts = grouped.groupby(

        "Genre"

    ).size()

    for genre, count in counts.items():

        print(

            f"{genre:<15}{count}"

        )

    print()

    print("Average Segments per Song")

    print("-" * 60)

    averages = grouped.groupby(

        "Genre"

    )["Segment_Count"].mean()

    for genre, avg in averages.items():

        print(

            f"{genre:<15}{avg:.2f}"

        )

    print()

    print(LINE_SEPARATOR)


# ==========================================================
# MAIN
# ==========================================================

def main():

    dataset = load_master_dataset()

    validate_dataset(

        dataset

    )

    grouped = group_songs(

        dataset

    )

    save_song_groups(

        grouped

    )

    generate_report(

        grouped

    )

    print_summary(

        grouped

    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()

    