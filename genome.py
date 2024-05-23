import dataclasses
import math
import random
import shutil
import typing

import numpy as np

from hyps import Hyps, MAX_HYPS, MIN_HYPS
from training import train as objective


class Genome:
    def __init__(self, hyps: Hyps = None):
        self.hyps = hyps if hyps else Hyps()
        self.lifetime = 0
        self.fitness = 0

    def set_lifetime(self, fitness, max_lifetime):
        self.fitness = fitness
        self.lifetime = int(max_lifetime * (math.pow(fitness, 1/4) + fitness)) + 2
        return self

    @staticmethod
    def crossover(a: "Genome", b: "Genome"):
        c = Genome(Hyps())
        for field in dataclasses.fields(Hyps):
            c.hyps.__setattr__(field.name, random.choice([
                a.hyps.__getattribute__(field.name),
                b.hyps.__getattribute__(field.name)
            ]))
        return c

    def mutate(self, p=0.2):
        for field in dataclasses.fields(Hyps):
            if random.random() > p:
                continue
            old_val = self.hyps.__getattribute__(field.name)
            max_val = MAX_HYPS.__getattribute__(field.name)
            min_val = MIN_HYPS.__getattribute__(field.name)
            if hasattr(field.type, "__args__"):
                choices = field.type.__args__
                new_val = random.choice(choices)
            elif field.type == int:
                max_change = int(math.ceil((max_val - min_val) * p))
                new_val = random.randint(
                    max(min_val, old_val - max_change),
                    min(max_val, old_val + max_change)
                )
            elif field.type == float:
                max_change = (max_val - min_val) * p
                new_val = random.uniform(
                    max(min_val, old_val - max_change),
                    min(max_val, old_val + max_change)
                )
            else:
                new_val = old_val
            self.hyps.__setattr__(field.name, new_val)
        return self


class Population:
    members = []

    def __init__(self, objective: typing.Callable[[Hyps], float], exp_size: int = 10):
        self.size = exp_size
        self.max_lifetime = self.size
        self.objective = objective
        self._generate()

    def _generate(self):
        self.members = [
            Genome.crossover(Genome(MIN_HYPS), Genome(MAX_HYPS)).mutate().set_lifetime(0, self.max_lifetime)
            for _ in range(self.size)
        ]

    def select(self) -> typing.Tuple[Genome, Genome]:
        # a = random.choice(self.members)
        # b = random.choice(self.members)
        sel = sorted(self.members, key=lambda x: x.fitness, reverse=True)[:2]
        if len(sel) == 1:
            return sel[0], sel[0]
        a, b = sel
        return a, b

    def evolution_step(self):
        parent_a, parent_b = self.select()
        child = Genome.crossover(parent_a, parent_b)
        child.mutate()
        fitness = self.objective(child.hyps)
        child.set_lifetime(fitness, self.max_lifetime)

        new_members = []
        for member in self.members:
            member.lifetime -= 1
            if member.lifetime > 0:
                new_members.append(member)
        self.members = new_members

        self.members.append(child)
        print(" ".join([f"{mem.fitness:.3f}" for mem in sorted(self.members, key=lambda x: x.fitness)]))

    def get_best(self):
        idx = np.argmax([mem.fitness for mem in self.members])
        return self.members[idx]

    def evolve(self, steps=10):
        for step in range(steps):
            self.evolution_step()
        return self.get_best()


if __name__ == '__main__':
    shutil.rmtree("lightning_logs", ignore_errors=True)
    pop = Population(exp_size=10, objective=objective)
    best = pop.evolve(1000)
    print(best.hyps)
