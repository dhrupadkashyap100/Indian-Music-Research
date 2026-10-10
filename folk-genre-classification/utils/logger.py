"""
=========================================================
Logger Utility
=========================================================
"""

from pathlib import Path
import csv
from datetime import datetime


class ProjectLogger:

    def __init__(self, log_file: Path):

        self.log_file = log_file

        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if not self.log_file.exists():

            with open(
                self.log_file,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "Timestamp",
                    "Level",
                    "Genre",
                    "Input File",
                    "Output File",
                    "Message"
                ])

    # ======================================================
    # Internal writer
    # ======================================================

    def write(
        self,
        level,
        genre,
        input_file,
        output_file,
        message
    ):

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with open(
            self.log_file,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                timestamp,
                level,
                genre,
                input_file,
                output_file,
                message
            ])

    # ======================================================
    # Generic logger (used by scripts)
    # ======================================================

    def log(
        self,
        genre,
        song,
        status,
        message
    ):

        self.write(
            level=status,
            genre=genre,
            input_file=song,
            output_file="",
            message=message
        )

    # ======================================================
    # INFO
    # ======================================================

    def info(
        self,
        genre,
        input_file,
        output_file,
        message
    ):

        self.write(
            "INFO",
            genre,
            input_file,
            output_file,
            message
        )

    # ======================================================
    # WARNING
    # ======================================================

    def warning(
        self,
        genre,
        input_file,
        output_file,
        message
    ):

        self.write(
            "WARNING",
            genre,
            input_file,
            output_file,
            message
        )

    # ======================================================
    # ERROR
    # ======================================================

    def error(
        self,
        genre,
        input_file,
        output_file,
        message
    ):

        self.write(
            "ERROR",
            genre,
            input_file,
            output_file,
            message
        )