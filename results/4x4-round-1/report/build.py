"""Build the page of tables and figures (report/matmulnet-4x4.html) from report/data.json.

    python results/4x4-round-1/report/build.py
"""
import collections, json, math, os
import numpy as np
T = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{T}/data.json"))
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
def p10(k): return "1" if k == 0 else "10" + str(k).translate(SUP)
def sci(x, d=1):
    if x == 0: return "0"
    k = math.floor(math.log10(abs(x))); m = x / 10 ** k
    if round(m, d) >= 10: m, k = m / 10, k + 1
    ms = f"{m:.{d}f}".replace(".", ",")
    return ms if k == 0 else f"{ms}×10{str(k).translate(SUP)}"
def dec(x, d=3): return f"{x:.{d}f}".replace(".", ",")

# ---------- SVG ----------
def chart(series, xr, yr, xlog=True, ylog=True, xticks=None, yticks=None, xlabel="", ylabel="", w=660, h=380,
          cats=None, xrev=False, hlines=(), xfmt=None, yfmt=None):
    L, R, Tp, B = 62, 34, 14, 46
    pw, ph = w - L - R, h - Tp - B
    fx = (lambda v: math.log10(v)) if xlog else (lambda v: v)
    fy = (lambda v: math.log10(v)) if ylog else (lambda v: v)
    x0, x1, y0, y1 = fx(xr[0]), fx(xr[1]), fy(yr[0]), fy(yr[1])
    def X(v):
        t = (fx(v) - x0) / (x1 - x0); t = 1 - t if xrev else t
        return L + t * pw
    def Y(v): return Tp + (1 - (fy(v) - y0) / (y1 - y0)) * ph
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" xmlns="http://www.w3.org/2000/svg">']
    for v in yticks:
        o.append(f'<line class="grid" x1="{L}" x2="{w-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        lab = yfmt(v) if yfmt else (p10(round(math.log10(v))) if ylog else str(v))
        o.append(f'<text class="tick" x="{L-8}" y="{Y(v)+4:.1f}" text-anchor="end">{lab}</text>')
    for v in xticks:
        o.append(f'<line class="axis" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{Tp+ph}" y2="{Tp+ph+5}"/>')
        lab = cats[v] if cats else (xfmt(v) if xfmt else (p10(round(math.log10(v))) if xlog else str(v)))
        o.append(f'<text class="tick" x="{X(v):.1f}" y="{Tp+ph+20}" text-anchor="middle">{lab}</text>')
    o.append(f'<line class="axis" x1="{L}" x2="{w-R}" y1="{Tp+ph}" y2="{Tp+ph}"/>')
    o.append(f'<text class="lab" x="{L+pw/2}" y="{h-6}" text-anchor="middle">{xlabel}</text>')
    o.append(f'<text class="lab" transform="translate(14 {Tp+ph/2}) rotate(-90)" text-anchor="middle">{ylabel}</text>')
    for v, cls, txt in hlines:
        o.append(f'<line class="ref {cls}" x1="{L}" x2="{w-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        o.append(f'<text class="tick" x="{w-R-4}" y="{Y(v)-5:.1f}" text-anchor="end">{txt}</text>')
    clipy = lambda v: min(max(v, yr[0]), yr[1]) if yr[0] < yr[1] else v
    for s in series:
        pts = [(x, y) for x, y in s["pts"] if y > 0 or not ylog]
        cls = s["cls"]
        if s.get("line", True) and len(pts) > 1:
            d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f} {Y(clipy(y)):.1f}" for i, (x, y) in enumerate(pts))
            o.append(f'<path class="ln {cls}{" dash" if s.get("dash") else ""}" d="{d}" fill="none"/>')
        if s.get("dots"):
            for x, y in pts:
                if yr[0] <= y <= yr[1]:
                    o.append(f'<circle class="dot {cls}" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="{s.get("r", 4)}"/>')
        for x, y, e in s.get("marks", []):       # mean marker with geometric spread
            o.append(f'<line class="mean" x1="{X(x)-11:.1f}" x2="{X(x)+11:.1f}" y1="{Y(y):.1f}" y2="{Y(y):.1f}"/>')
    o.append("</svg>")
    return "\n".join(o)

def legend(items): return '<div class="legend">' + "".join(f'<span><i class="sw {c}{" dash" if d else ""}"></i>{t}</span>' for c, t, d in items) + "</div>"
def fig(n, title, svg, leg, cap): return f'<figure id="fig{n}"><figcaption><b>Figura {n}.</b> {title}</figcaption><div class="scroll">{svg}</div>{leg}<p class="cap">{cap}</p></figure>'
def table(n, title, head, rows, cap="", cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<figure id="tab{n}"><figcaption><b>Tabella {n}.</b> {title}</figcaption><div class="scroll"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>' + (f'<p class="cap">{cap}</p>' if cap else "") + "</figure>"

rank = collections.defaultdict(list)
for x in D["rank"]: rank[x["r"]].append(x)
border = collections.defaultdict(list)
for x in D["border"]: border[x["r"]].append(x)
jit = [-0.24, -0.16, -0.08, 0.0, 0.08, 0.16, 0.24]

# Fig 1: rank strip
rs = sorted(rank)
ser = []
for i, r in enumerate(rs):
    xs = sorted(rank[r], key=lambda v: v["seed"])
    cls = "c1" if r <= 49 else "c2"
    ser.append(dict(pts=[(i + (jit[x["seed"] - 1] if len(xs) > 1 else 0), x["loss"]) for x in xs], cls=cls, line=False, dots=True,
                    marks=[(i, float(np.mean([x["loss"] for x in xs])), 0)] if len(xs) > 1 else []))
f1 = chart(ser, (-0.6, len(rs) - 0.4), (1e-5, 2e-1), xlog=False, xticks=list(range(len(rs))), cats={i: str(r) for i, r in enumerate(rs)},
           yticks=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], xlabel="numero di termini r", ylabel="loss di validazione (migliore)")
l1 = legend([("c1", "r = 45…49, sette seed per rango", 0), ("c2", "r = 50…64, un seed per rango", 0), ("meansw", "media sui sette seed", 0)])

# Fig 2: rank curves
ramp = {45: "a1", 46: "a2", 47: "a3", 48: "a4", 49: "a5", 51: "b1", 54: "b2", 60: "b3", 64: "b4"}
ser = [dict(pts=D["rank_curves"][str(r)], cls=ramp[r]) for r in ramp]
f2 = chart(ser, (100, 1e6), (1e-5, 1), xticks=[1e2, 1e3, 1e4, 1e5, 1e6], yticks=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], xlabel="epoca", ylabel="loss di validazione")
l2 = legend([(ramp[r], f"r = {r}" + (" (mediana)" if r <= 49 else ""), 0) for r in ramp])

# Fig 3: border strip
bs = sorted(border)
ser = [dict(pts=[(i + jit[x["seed"] - 1], x["loss"]) for x in border[r]], cls="c1", line=False, dots=True,
            marks=[(i, float(np.mean([x["loss"] for x in border[r]])), 0)]) for i, r in enumerate(bs)]
ser += [dict(pts=[(i + 0.38, float(np.mean([x["loss"] for x in rank[r]])))], cls="c2", line=False, dots=True, r=5) for i, r in enumerate(bs) if r in rank]
f3 = chart(ser, (-0.6, 3.6), (2e-2, 1e-1), xlog=False, xticks=[0, 1, 2, 3], cats={i: str(r) for i, r in enumerate(bs)},
           yticks=[2e-2, 3e-2, 5e-2, 7e-2, 1e-1], yfmt=lambda v: sci(v, 0), xlabel="numero di termini r", ylabel="loss di validazione (migliore)", h=340)
l3 = legend([("c1", "modello di bordo, sette seed per rango", 0), ("meansw", "media sui sette seed", 0), ("c2", "media del modello di rango allo stesso r", 0)])

# Fig 4: border loss along schedule
bc = {44: "a1", 45: "a2", 46: "a4", 47: "a5"}
ser = [dict(pts=D["border_curves"][str(r)], cls=bc[r]) for r in bc]
ser.append(dict(pts=D["smirnov_ctrl"], cls="c3"))
ser.append(dict(pts=[(e, l) for e, rho, l in D["smirnov_exact"] if e >= 1e-3], cls="c4", dash=True))
f4 = chart(ser, (1e-1, 1e-3), (1e-7, 1e8), xticks=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3], xfmt=lambda v: sci(v, 0),
           yticks=[1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8], xlabel="ε durante il training (decresce verso destra)", ylabel="loss di validazione", h=400)
l4 = legend([(bc[r], f"r = {r} (mediana di sette seed)", 0) for r in bc] + [("c3", "training avviato dalla famiglia di Smirnov, r = 46", 0), ("c4", "famiglia di Smirnov a coefficienti fissi", 1)])

# Fig 5: fixed coefficients, rho vs eps
ser = []
for r in bc:
    eps = [e for e, _ in border[r][0]["at"]]
    med = np.median(np.array([[v for _, v in x["at"]] for x in border[r]]), 0)
    ser.append(dict(pts=list(zip(eps, [float(m) for m in med])), cls=bc[r], dots=True, r=3))
ser.append(dict(pts=[(e, rho) for e, rho, l in D["smirnov_exact"]], cls="c4", dash=True))
f5 = chart(ser, (1e-1, 1e-6), (1e-7, 1e17), xticks=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
           yticks=[1e-6, 1e-3, 1, 1e3, 1e6, 1e9, 1e12, 1e15], xlabel="ε a cui si valutano i coefficienti del best (decresce verso destra)", ylabel="residuo relativo ρ", h=400)
l5 = legend([(bc[r], f"r = {r} (mediana di sette best)", 0) for r in bc] + [("c4", "famiglia di Smirnov, r = 46", 1)])

# Fig 6: refinement rho vs W
rc = {(47, 1): "a2", (48, 4): "a4", (49, 6): "a5", (49, 1): "c3", (51, 1): "b1", (52, 1): "b2", (54, 1): "b3", (60, 1): "b4"}
ser = []
for x in sorted(D["refine"], key=lambda v: (v["r"], v["seed"])):
    pts = [(x["start"]["W"], x["start"]["rho"])] + [(l["W"], l["rho"]) for l in x["levels"]]
    ser.append(dict(pts=pts, cls=rc[(x["r"], x["seed"])], dots=True, r=3.5))
f6 = chart(ser, (1e3, 3e5), (1e-6, 1), xticks=[1e3, 1e4, 1e5], yticks=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
           xlabel="peso W = Σ ‖h‖‖k‖‖f‖", ylabel="residuo relativo ρ", h=400,
           hlines=[(0.25, "", "√(4/64)"), (math.sqrt(2 / 64), "", "√(2/64)")])
l6 = legend([(rc[k], f"r = {k[0]}, seed {k[1]}", 0) for k in sorted(rc)])

# ---------- tables ----------
def stats(xs, key="loss"):
    a = np.array([x[key] for x in xs]); return a.mean(), a.std(ddof=1), a.min(), a.max()
t_rank = [[r, *(sci(v) for v in stats(rank[r])), dec(float(np.median([x["rho"] for x in rank[r]]))), dec(min(x["rho"] for x in rank[r])),
           f"0/{len(rank[r])}"] for r in range(45, 50)]
t_rank_runs = [[r] + [sci(x["loss"]) for x in sorted(rank[r], key=lambda v: v["seed"])] for r in range(45, 50)]
t_ladder = [[r, sci(rank[r][0]["loss"]), dec(rank[r][0]["rho"]), sci(rank[r][0]["W"])] for r in rs if r >= 50]
def med(xs, k): return float(np.median([x[k] for x in xs]))
t_border = [[r, *(sci(v) for v in stats(border[r])), dec(med(border[r], "eps")), dec(med(border[r], "rho_tr")), dec(med(border[r], "neg"), 1),
             f'{med(border[r], "t0"):.0f}', sci(med(border[r], "last"))] for r in bs]
t_border_runs = [[r] + [f'{sci(x["loss"])}<br><small>ε = {dec(x["eps"])}</small>' for x in sorted(border[r], key=lambda v: v["seed"])] for r in bs]
epsl = [1e-1, 1e-2, 1e-3, 1e-4]
t_lim = []
for r in bs:
    row = [f"r = {r}"]
    for e in epsl:
        row.append(sci(float(np.median([dict(x["at"])[e] for x in border[r]]))))
    t_lim.append(row)
t_lim.append(["Smirnov, r = 46"] + [sci(min(D["smirnov_exact"], key=lambda v: abs(math.log10(v[0] / e)))[1]) for e in epsl])
t_ref = []
for x in sorted(D["refine"], key=lambda v: (v["r"], v["seed"])):
    row = [x["r"], x["seed"], f'{sci(x["start"]["rho"])}<br><small>W = {sci(x["start"]["W"])}</small>']
    for l in x["levels"]:
        row.append(f'{sci(l["rho"], 2)}<br><small>c = {l["c"]:g}, W = {sci(l["W"])}</small>')
    t_ref.append(row)
names = {"strassen49": "Strassen ⊗ Strassen, 49 termini", "rational48": "Dumas–Pernet–Sedoglavic, 48 termini"}
t_ctrl = []
for k in ("strassen49", "rational48"):
    tr = {x["name"].rsplit("_s", 1)[1].split("_")[0]: x for x in D["ctrl_train"] if k in x["name"]}
    ch = D["chain"][k][-1]; dr = D["direct_ctrl"][k][-1]
    t_ctrl.append([names[k], "0", " · ".join(sci(tr[s]["best_val_loss"]) for s in ("1e-3", "1e-2", "1e-1")),
                   f'{sci(dr["rho"])} <small>({dr["it"]} iterazioni)</small>', f'{dec(ch["rho"], 4)} <small>(W = {sci(ch["W"])})</small>'])
ss = D["smirnov_sum"]
t_chain = [[f'{l["c"]:g}', dec(D["chain"]["strassen49"][i]["rho"], 4), sci(D["chain"]["strassen49"][i]["W"]), dec(D["chain"]["rational48"][i]["rho"], 4), sci(D["chain"]["rational48"][i]["W"])]
           for i, l in enumerate(D["chain"]["strassen49"]) if l["c"] in (0.1, 0.3, 0.5, 0.7, 1, 2, 5, 10, 25)]
hr = lambda xs: float(np.mean([x["sec"] for x in xs])) / 3600
rank_h = hr([x for x in D["rank"]]); bord_h = hr(D["border"])
t_time = [["rango, r = 45…64", 42, dec(rank_h * 3.6, 1), dec(rank_h, 1), "313.000.000"], ["bordo, r = 44…47", 28, dec(bord_h * 3.6, 1), dec(bord_h, 1), "313.000.000"]]

CSS = open(f"{T}/style.css").read()
H = f'''<title>MatMulNet 4×4</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Serif:wght@500&display=swap">
<style>{CSS}</style>
<main>
<header>
<p class="eyebrow">MatMulNet · esperimenti del 9–12 ottobre 2026</p>
<h1>Moltiplicazione di matrici 4×4: tabelle e figure</h1>
<p class="lede">Training, verifica del limite in ε e raffinamento vincolato sul tensore M<sub>4</sub>, con il protocollo degli esperimenti 3×3. Riferimenti noti: 34 ≤ R(M<sub>4</sub>) ≤ 48 e 29 ≤ <u>R</u>(M<sub>4</sub>) ≤ 46.</p>
<nav><a href="#sec-setup">Protocollo</a><a href="#sec-rank">Rango</a><a href="#sec-border">Rango di bordo</a><a href="#sec-ref">Raffinamento</a></nav>
</header>

<section id="sec-setup"><h2>Protocollo e controlli</h2>
{table(1, "Protocollo comune a tutte le run.", ["voce", "valore"], [
 ["dati", "10.000 triple di training e 1.000 di validazione (A, B, AB), entrate uniformi in [−1, 1], estratte una volta"],
 ["ottimizzazione", "Adam, batch 32, norma del gradiente limitata a 10; 313 aggiornamenti per epoca"],
 ["learning rate", "10⁻⁴, dimezzato sui plateau della loss di validazione (pazienza 15), minimo 10⁻⁶"],
 ["durata", "10⁶ epoche per run"],
 ["inizializzazione", "rango: N(0, 1). Bordo: coefficiente di ε<sup>e</sup> da N(0, 1) diviso per 0,1<sup>e</sup>"],
 ["modello di bordo", "H, K, F polinomi di Laurent in ε con esponenti da −1 a 2; ε da 10⁻¹ a 10⁻³ con decadimento esponenziale"],
 ["aritmetica", "doppia precisione"],
 ["quantità riportate", "miglior loss di validazione; ρ = ‖T − M<sub>4</sub>‖ / ‖M<sub>4</sub>‖ su tutte le 4096 entrate; W = Σ<sub>s</sub> ‖h<sub>s</sub>‖‖k<sub>s</sub>‖‖f<sub>s</sub>‖"],
 ["relazione loss–residuo", "per matrici 4×4 con entrate uniformi in [−1, 1] la loss attesa è 4ρ²/9"]], cls="kv")}
{table(2, "Schemi noti usati come controllo: residuo dello schema, training avviato dallo schema perturbato, raffinamento.",
 ["schema", "ρ dello schema", "loss migliore dopo 20.000 epoche<br><small>rumore relativo 10⁻³ · 10⁻² · 10⁻¹</small>", "raffinamento con il solo vincolo c = 2<br><small>dallo schema perturbato dell'1%</small>", "catena completa da c = 0,1 a c = 25<br><small>dallo schema perturbato dell'1%</small>"],
 t_ctrl + [["Smirnov, 46 termini, esponenti −1…2", "0,76·ε<br><small>gradi negativi nulli, grado zero = M<sub>4</sub></small>",
            f'{sci(ss["best_val_loss"])} a ε = {dec(ss["epsilon_at_best"])}, poi {sci(ss["last_val_loss"])} a fine run<br><small>10⁶ epoche, senza rumore</small>', "—", "—"]],
 "Fonti: catalogo fmm.univ-lille.fr (48 termini, arXiv:2506.13242) e arXiv:1412.1687 (46 termini). Lo schema a 49 termini è il prodotto di Kronecker dello schema di Strassen con sé stesso.")}
{table(3, "Tempi di calcolo, un core per run.", ["sweep", "run", "ms per epoca", "ore per run", "aggiornamenti per run"], t_time)}
</section>

<section id="sec-rank"><h2>Rango</h2>
{fig(1, "Miglior loss di validazione per run, modello di rango.", f1, l1, "Ogni punto è una run da 10⁶ epoche. La soglia di successo usata nel 3×3 è 10⁻³.")}
{table(4, "Modello di rango, r = 45…49: statistiche su sette seed.", ["r", "media", "dev. std.", "minimo", "massimo", "ρ mediano", "ρ minimo", "run sotto 10⁻³"], t_rank)}
{table(5, "Modello di rango, r = 45…49: miglior loss di validazione per seed.", ["r"] + [f"seed {i}" for i in range(1, 8)], t_rank_runs)}
{table(6, "Modello di rango, r = 50…64: un seed per rango.", ["r", "loss migliore", "ρ", "W"], t_ladder)}
{fig(2, "Loss di validazione durante il training, modello di rango.", f2, l2, "Per r = 45…49 la curva è la mediana dei sette seed; per gli altri ranghi è il seed 1. Il learning rate raggiunge il minimo 10⁻⁶ entro le prime 10.000 epoche in tutte le run.")}
</section>

<section id="sec-border"><h2>Rango di bordo</h2>
{fig(3, "Miglior loss di validazione per run, modello di bordo.", f3, l3, "Il best di ogni run è raggiunto a ε compreso fra 0,024 e 0,084.")}
{table(7, "Modello di bordo, r = 44…47: statistiche su sette seed e verifica del limite sui best.",
 ["r", "media", "dev. std.", "minimo", "massimo", "ε al best<br><small>mediana</small>", "ρ a quell'ε<br><small>mediana</small>", "parte negativa<br><small>mediana</small>", "‖T<sub>0</sub> − M<sub>4</sub>‖ / ‖M<sub>4</sub>‖<br><small>mediana</small>", "loss a fine run<br><small>mediana</small>"], t_border,
 "Con i coefficienti del best tenuti fissi il tensore è T(ε) = Σ<sub>d</sub> ε<sup>d</sup> T<sub>d</sub>. La parte negativa è la norma dei T<sub>d</sub> con d &lt; 0 divisa per ‖M<sub>4</sub>‖. La famiglia converge a M<sub>4</sub> solo se la parte negativa è zero e T<sub>0</sub> = M<sub>4</sub>; per la famiglia di Smirnov entrambe le colonne valgono 0.")}
{table(8, "Modello di bordo: miglior loss di validazione per seed, con l'ε a cui è raggiunta.", ["r"] + [f"seed {i}" for i in range(1, 8)], t_border_runs)}
{fig(4, "Loss di validazione lungo lo schedule di ε.", f4, l4, "La linea tratteggiata è la loss 4ρ²/9 della famiglia di Smirnov valutata allo stesso ε senza alcun training.")}
{fig(5, "Residuo ρ dei best valutati a ε decrescente, a coefficienti fissi.", f5, l5, "Ogni punto è la mediana sui sette best del rango. Per una degenerazione di bordo ρ tende a zero con ε, come per la famiglia di Smirnov.")}
{table(9, "Residuo ρ a coefficienti fissi, per alcuni valori di ε (mediana sui sette best).", ["", "ε = 10⁻¹", "ε = 10⁻²", "ε = 10⁻³", "ε = 10⁻⁴"], t_lim)}
</section>

<section id="sec-ref"><h2>Raffinamento vincolato</h2>
{fig(6, "Residuo contro peso lungo il raffinamento dei candidati di rango.", f6, l6, "Il primo punto di ogni linea è il best del training; i successivi sono i tre livelli del vincolo |θ| ≤ c. Le due linee orizzontali sono i residui di un tensore che differisce da M<sub>4</sub> per 4 e per 2 entrate.")}
{table(10, "Raffinamento diretto dei candidati di rango (Levenberg–Marquardt proiettato, al più un'ora per livello).", ["r", "seed", "best del training", "primo livello", "secondo livello", "terzo livello"], t_ref,
 "Il primo livello è il più piccolo c che non taglia alcun coefficiente del candidato. Per r = 49 seed 1 e r = 52 il tetto era di 20 minuti per livello. Quasi tutti i livelli si sono fermati per il tetto di tempo.")}
{table(11, "Catena completa (c da 0,1 a 25) avviata dagli schemi noti perturbati dell'1%.", ["c", "49 termini: ρ", "49 termini: W", "48 termini: ρ", "48 termini: W"], t_chain,
 "Valori di riferimento: √(3/64) = 0,2165 e √(4/64) = 0,2500.")}
</section>
<footer>Dati per run (fattori al best e a fine run, curve per epoca, verifiche del limite) nel progetto W&amp;B <span class="mono">matmulnet-4x4</span> e nella cartella <span class="mono">runs/</span> del codice.</footer>
</main>'''
open(f"{T}/matmulnet-4x4.html", "w").write(H)
print(len(H))
