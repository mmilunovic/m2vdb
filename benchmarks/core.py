"""Lightweight benchmarking harness for m2vdb experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List

from .utils import BenchmarkRunner


@dataclass
class BenchmarkCase:
    name: str
    description: str
    func: Callable[[BenchmarkRunner], BenchmarkRunner | None]

    def run(self) -> BenchmarkRunner:
        runner = BenchmarkRunner(self.name, self.description)
        result = self.func(runner)
        return result if isinstance(result, BenchmarkRunner) else runner


@dataclass
class BenchmarkReport:
    runners: List[BenchmarkRunner] = field(default_factory=list)

    def to_markdown(self) -> str:
        sections = []
        for runner in self.runners:
            table = runner.to_markdown()
            if table:
                sections.append(f"### {runner.name}\n\n{table}")
        return "\n\n".join(sections)

    def print(self) -> None:
        for runner in self.runners:
            runner.print_results()


class BenchmarkSuite:
    def __init__(self, cases: Iterable[BenchmarkCase]):
        self.cases = list(cases)

    def run(self) -> BenchmarkReport:
        runners = [case.run() for case in self.cases]
        return BenchmarkReport(runners)
