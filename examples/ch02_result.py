# replay: exact
"""Chapter 2: from indications to a result."""
from swmetro import Budget, Result
from swmetro import type_a, u_resolution

# Ten indications, in ms, of the duration of one
# cache lookup under repeatability conditions.
# The timer displays steps of 0.01 ms.
IND = [12.31, 12.28, 12.35, 12.30, 12.27,
       12.33, 12.29, 12.36, 12.32, 12.30]
STEP = 0.01


def main():
    a = type_a(IND, "ms")
    bud = Budget("ms")
    bud.add("repeatability", "A", a.u, a.dof)
    bud.add("timer resolution", "B",
            u_resolution(STEP))
    u_c, dof = bud.u_c(), bud.dof()
    r = Result(a.value, u_c, "ms", dof)
    print(bud.table())
    print(f"effective dof {r.dof:.1f}")
    print("with standard uncertainty:")
    print("  ", r)
    print("with expanded uncertainty:")
    print("  ", r.report())


if __name__ == "__main__":
    main()
