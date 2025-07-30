import random
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.stats import norm
import os


@dataclass
class CourseInfo:
    code: str
    name: str
    description: str


class PreferenceEngine:
    def __init__(self, courses: List[CourseInfo]):
        self.courses = courses
        self.n = len(courses)
        self.mu = np.zeros(self.n)
        self.sigma = np.ones(self.n)
        self.comparisons = []  # list of (winner_idx, loser_idx)

    def _prob_win(self, i, j):
        return norm.cdf((self.mu[i] - self.mu[j]) / np.sqrt(2))

    def _update_estimates(self):
        learning_rate = 0.1
        for _ in range(10):
            gradient = np.zeros(self.n)
            for winner, loser in self.comparisons:
                diff = self.mu[winner] - self.mu[loser]
                p = norm.cdf(diff / np.sqrt(2))
                if p in (0, 1):
                    continue
                dL = norm.pdf(diff / np.sqrt(2)) / (p * np.sqrt(2))
                gradient[winner] += dL
                gradient[loser] -= dL
            self.mu += learning_rate * gradient

    def _expected_info_gain(self, i, j):
        p_win = self._prob_win(i, j)
        return -p_win * np.log(p_win + 1e-8) - (1 - p_win) * np.log(1 - p_win + 1e-8)

    def choose_next_pair(self):
        best_pair = None
        best_gain = -float("inf")
        for i in range(self.n):
            for j in range(i + 1, self.n):
                gain = self._expected_info_gain(i, j)
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (i, j)
        return best_pair

    def present_pair(self, i, j):
        def _display():
            os.system("cls" if os.name == "nt" else "clear")
            print("=" * 70)
            print("Which course do you prefer?\n")
            print(f"[1] {self.courses[i].code}: {self.courses[i].name}")
            print(f"     {self.courses[i].description}\n")
            print(f"[2] {self.courses[j].code}: {self.courses[j].name}")
            print(f"     {self.courses[j].description}\n")
            print("-" * 70)
            print(
                "Commands: [1] Choose first  [2] Choose second  [r] Show rankings  [q] Quit"
            )

        while True:
            _display()
            choice = input("Enter your choice: ").strip().lower()
            if choice in ("1", "2"):
                winner = i if choice == "1" else j
                loser = j if choice == "1" else i
                self.comparisons.append((winner, loser))
                self._update_estimates()
                return True  # continue
            elif choice == "r":
                self.show_rankings()
            elif choice == "q":
                return False  # stop
            else:
                print("Invalid input. Enter 1, 2, r, or q.")

    def show_rankings(self, top_n: int = None):
        os.system("cls" if os.name == "nt" else "clear")
        rankings = list(enumerate(self.mu))
        rankings.sort(key=lambda x: -x[1])
        print("\nCurrent Rankings:")
        print("=" * 40)
        for idx, score in rankings[: top_n or len(rankings)]:
            print(
                f"{self.courses[idx].code:<8} ({score:+.2f})  {self.courses[idx].name}"
            )
        print("=" * 40 + "\n")
        input("Press Enter to return to the next comparison...")


def main():
    with open("data.txt") as f:
        content = f.read()
        content = content.split("\n")
        content = [element.strip() for element in content if not element.strip() == ""]
        content = [content[i : i + 3] for i in range(0, len(content), 3)]

    courses = []
    for code, name, description in content:
        courses.append(CourseInfo(code=code, name=name, description=description))

    engine = PreferenceEngine(courses)

    while True:
        pair = engine.choose_next_pair()
        if pair is None:
            print("No more comparisons available.")
            break
        should_continue = engine.present_pair(*pair)
        if not should_continue:
            break

    print("\nFinal rankings:")
    engine.show_rankings()


if __name__ == "__main__":
    main()
