import json
from math import comb
from pathlib import Path

def upper_remaining_error_bound(population, checked, mismatches, alpha=0.05):
    if not 0 <= mismatches <= checked <= population:
        raise ValueError("Comptages invalides.")
    remaining = population-checked
    if remaining == 0:
        return 0.0

    def cumulative_probability(total_errors):
        denominator = comb(population, checked)
        numerator = 0
        for observed_errors in range(mismatches+1):
            correct_draws = checked-observed_errors
            if (
                0 <= observed_errors <= total_errors
                and 0 <= correct_draws <= population-total_errors
            ):
                numerator += (
                    comb(total_errors, observed_errors)
                    * comb(population-total_errors, correct_draws)
                )
        return numerator/denominator

    compatible_totals = [
        errors
        for errors in range(mismatches, mismatches+remaining+1)
        if cumulative_probability(errors) >= alpha
    ]
    maximum = max(compatible_totals, default=mismatches)
    return (maximum-mismatches)/remaining

def main():
    output = Path("autonomous_lab/runs/history_reliability_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)

    rows = []
    for checked in (0, 4, 8, 16, 24, 31):
        for mismatches in range(checked+1):
            upper = upper_remaining_error_bound(
                31, checked, mismatches
            )
            rows.append({
                "checked": checked,
                "mismatches": mismatches,
                "remaining": 31-checked,
                "upper_remaining_error_fraction": upper,
            })

        for mismatches in sorted(set((0, min(1, checked), checked))):
            upper = upper_remaining_error_bound(
                31, checked, mismatches
            )
            print(
                f"Controles={checked} | erreurs={mismatches} | "
                f"borne superieure des erreurs restantes={100*upper:.1f}%"
            )

    # Tests exhaustifs de couverture du calcul sur un domaine plus petit.
    # Pour chaque nombre reel d'erreurs, compter tous les echantillons
    # possibles via leur probabilite hypergeometrique.
    worst_failure = 0.0
    for population in range(1, 13):
        for checked in range(population):
            for errors in range(population+1):
                failure = 0.0
                for mismatches in range(checked+1):
                    correct_draws = checked-mismatches
                    if not (
                        mismatches <= errors
                        and 0 <= correct_draws <= population-errors
                    ):
                        continue
                    probability = (
                        comb(errors, mismatches)
                        * comb(population-errors, correct_draws)
                        / comb(population, checked)
                    )
                    actual = (errors-mismatches)/(population-checked)
                    bound = upper_remaining_error_bound(
                        population, checked, mismatches
                    )
                    if actual > bound + 1e-12:
                        failure += probability
                worst_failure = max(worst_failure, failure)

    assert worst_failure <= 0.05 + 1e-12
    print(
        "Controle mathematique : probabilite maximale de depassement "
        f"sur les petits domaines = {100*worst_failure:.3f}%"
    )

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "population": 31,
                "alpha": 0.05,
                "sampling": "uniform_without_replacement",
                "fixed_budget_bound": True,
                "noise": False,
                "stationary_during_checks": True,
            },
            "coverage_check": {
                "maximum_population_tested": 12,
                "worst_failure_probability": worst_failure,
            },
            "bounds": rows,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Resultats : {path}")

if __name__ == "__main__":
    main()
