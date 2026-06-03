import json
import re

text = r'''{
    "pembahasan": "Model matematis.\n\nUntuk memprediksi $$J_n = 2^n$$\\n\n**Keterbatasan:**\n1. Algoritma\n2. Intervensi\n\frac{1}{2} \times \left( x \right)"
}'''

print("ORIGINAL TEXT:")
print(repr(text))

# Let's see if we only replace backslashes for specific LaTeX commands
latex_commands = [
    "frac", "times", "left", "right", "sqrt", "alpha", "beta", "gamma", "delta", "pi", "theta", "mu", "nu", "rho", "sigma", "phi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi", "Omega", "sum", "prod", "int", "oint", "lim", "sin", "cos", "tan", "cot",
    "sec", "csc", "log", "ln", "exp", "max", "min", "to", "rightarrow", "leftarrow", "Rightarrow", "Leftarrow", "cdot", "infty", "approx", "neq", "leq",
    "geq", "equiv", "propto", "circ", "partial", "nabla", "textbf", "textit", "underline", "mathrm", "mathbf", "mathcal", "mathbb", "mathscr", "mathfrak",
    "text", "begin", "end", "hline", "vspace", "hspace", "rule", "color", "textcolor", "colorbox", "fcolorbox", "pagecolor", "usepackage", "newcommand",
    "renewcommand", "newenvironment", "renewenvironment", "newtheorem", "includegraphics", "caption", "label", "ref", "pageref", "cite", "bibliography",
    "bibliographystyle", "tableofcontents", "listoffigures", "listoftables", "part", "chapter", "section", "subsection", "subsubsection", "paragraph",
    "subparagraph", "item", "bmatrix", "pmatrix", "vmatrix", "Bmatrix", "Vmatrix", "cases", "align", "aligned", "gather", "gathered", "multline", "eqnarray",
    "array", "tabular", "table", "figure", "minipage", "center", "flushleft", "flushright", "itemize", "enumerate", "description", "maketitle", "author",
    "title", "date", "abstract", "appendix", "frontmatter", "mainmatter", "backmatter", "printindex", "printglossary", "printbibliography", "addcontentsline",
    "addtocontents", "markboth", "markright", "pagestyle", "thispagestyle", "pagenumbering", "setcounter", "addtocounter", "stepcounter", "refstepcounter",
    "arabic", "roman", "Roman", "alph", "Alph", "fnsymbol"
]

# Create a regex pattern that matches \(command) but only if it's preceded by a non-backslash
# Wait, if we use a regex like r'(?<!\\)\\(' + '|'.join(latex_commands) + r')\b'
pattern = r'(?<!\\)\\(' + '|'.join(latex_commands) + r')\b'
fixed_text = re.sub(pattern, r'\\\\\1', text)

print("\nFIXED TEXT:")
print(repr(fixed_text))

try:
    data = json.loads(fixed_text)
    print("\nPARSED JSON:")
    print(repr(data['pembahasan']))
except Exception as e:
    print("JSON Error:", e)
