"""Small authored diagnostic dataset, not a general assistant corpus."""
import json
from pathlib import Path


def pair(question, answer):
    return {"messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]}


def build(numbers):
    rows = []
    for n in numbers:
        rows.extend([
            pair(f"Calcule {n} + 2. Reponds uniquement avec le nombre.", str(n+2)),
            pair(f"Recopie uniquement le texte entre crochets : [bonjour {n}].", f"bonjour {n}"),
            pair(f"Extrais uniquement le nombre : objet numero {n}.", str(n)),
            pair(f"Le nombre {n} est-il pair ? Reponds par oui ou non.", "oui" if n % 2 == 0 else "non"),
        ])
        rows.append({"messages": [
            {"role": "user", "content": f"Retiens ce code : {n}. Reponds seulement OK."},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "Quel code viens-je de donner ? Reponds seulement avec ce code."},
            {"role": "assistant", "content": str(n)},
        ]})
    return rows


def main():
    root = Path("instructions_demo")
    root.mkdir(exist_ok=True)
    datasets = {"train": build(range(16)), "test": build(range(30, 34))}
    paths = [root/f"{split}.jsonl" for split in datasets]
    if any(p.exists() for p in paths):
        raise SystemExit("Donnees demo deja presentes : aucun remplacement effectue.")
    for split, rows in datasets.items():
        path = root/f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
        print(f"{path} : {len(rows)} exemples")
    print("Jeu synthetique de diagnostic : nombres de test distincts, patrons partages.")


if __name__ == "__main__":
    main()
