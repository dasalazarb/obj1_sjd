"""AST guard for the closed presentation-only operation policy."""
import ast
from pathlib import Path
import sys

BANNED_IMPORTS={'scipy','statsmodels','sklearn','lifelines','pingouin','numpy'}
BANNED_CALLS={'mean','median','std','quantile','corr','groupby','agg','aggregate','rolling','merge','join','fillna','interpolate','resample','sum','cumsum','fit','predict','read_parquet','read_pickle','eval','exec'}


def violations(root):
    errors=[]
    for area in ['pages','objectives','studies','shared']:
        folder=Path(root)/area
        for file in folder.rglob('*.py'):
            tree=ast.parse(file.read_text())
            for node in ast.walk(tree):
                modules=[]
                if isinstance(node,ast.Import): modules=[alias.name for alias in node.names]
                if isinstance(node,ast.ImportFrom) and node.module: modules=[node.module]
                if any(module.split('.')[0] in BANNED_IMPORTS for module in modules):
                    errors.append(f'{file}:{node.lineno}: forbidden import')
                if isinstance(node,ast.Call):
                    name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                    if name=='join' and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Constant) and isinstance(node.func.value.value,str):
                        continue # Literal string formatting, not a result-table join.
                    if name in BANNED_CALLS: errors.append(f'{file}:{node.lineno}: forbidden call {name}')
    return errors


if __name__=='__main__':
    errors=violations(Path(__file__).resolve().parents[1])
    if errors: print('\n'.join(errors)); sys.exit(1)
    print('Presentation-only AST checks passed')
