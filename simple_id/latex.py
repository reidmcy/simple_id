import re
import pandas

def latexPostProcessing(s, add_midrule = False):

    splitLines = s.split('\n')
    retS = '\n'.join(splitLines[:4])
    retLines = []
    for line in splitLines[4:-2]:
        retLines.append(re.sub(r'(^|\s|-)\d+\.?\d*', lambda x : "\\num{{{}}}".format(x.group(0).strip()), line))
    if add_midrule:
        retLines.insert(-2, '\\midrule')
    retS += '\n' + '\n'.join(retLines)
    return retS + '\n' + '\n'.join(splitLines[-2:])

def makeTable(df, caption = '', label = 't1', add_midrule = False):
    #df = df.copy()
    #df.columns = [c for c in df.columns]
    print("\\begin{table}[h]")
    print("     \centering")
    print("     \\begin{adjustbox}{center}")
    print(latexPostProcessing(df.to_latex(), add_midrule = add_midrule).strip())
    print("     \\end{adjustbox}")
    print("     \caption{{{}}}\label{{{}}}".format(caption, label))
    print("\end{table}")

texNames = [
    ('wos_id', 'ID'),
    ('is_comp', 'Explicitly Computational'),
    ('probPos', 'Likelyhood is Computational'),
    ('source' , 'Source'),
    ('pubyear' , 'Year of Publications'),
    ('title' , 'Title'),
    ('abstract' , 'Abstract'),
    ]

def rowToTex(row, cutoff = 60):
    print(r"""\begin{figure}[H]
	\begin{tabular}{ll}
		\toprule
		Field & Value\\
		\midrule""")
    for rN, tN in texNames:
        if rN == 'probPos':
             print('\t\t{} & {:.1f}\% \\\\'.format(tN, row[rN] * 100))
        elif len(str(row[rN])) < cutoff:
            print('\t\t{} & {} \\\\'.format(tN, row[rN]))
        else:
            s = str(row[rN])
            ts = s.split(' ')
            sOut = ['']
            while len(ts) > 0:
                subT = ts.pop(0)
                if len(sOut[-1] + ' ' + subT) < cutoff:
                    sOut[-1] += ' ' + subT
                else:
                    sOut.append(subT)
            print('\t\t{} & {} \\\\'.format(tN, '\\\\\n\t\t&'.join(sOut)))
    print(r"""		\bottomrule
	\end{tabular}
\end{figure}""")
