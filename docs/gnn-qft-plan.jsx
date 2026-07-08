import { useState } from "react";

const phases = [
  {
    id: 1,
    label: "Phase I",
    title: "Scalar Field Theory on a Fixed Lattice",
    subtitle: "Euclidean φ⁴ in 2D",
    status: "Foundation",
    color: "#00d4ff",
    duration: "~3 months",
    goal: "Prove the heterogeneous GNN architecture can reproduce known φ⁴ results — correlation lengths, critical exponents, phase transitions — matching analytic and Monte Carlo benchmarks.",
    theory: {
      lagrangian: "ℒ_E = ½(∂_μφ)² + ½m²φ² + λφ⁴",
      signature: "Euclidean (+ + + +)",
      dimension: "2D lattice (N×N sites)",
      rationale: "2D φ⁴ is exactly solvable near criticality (Ising universality class). Benchmarks exist. No fermions. No gauge fields. Minimal complexity."
    },
    graph: {
      spacetime_nodes: "N² sites, features: [x¹, x², lattice_spacing a]",
      field_nodes: "One φ(x) node per site, features: [field_value ∈ ℝ]",
      st_st_edges: "4-connected grid (±x̂, ±ŷ), edge feature: direction μ",
      st_field_edges: "Bipartite inhabits edges (each site ↔ its φ node)"
    },
    architecture: [
      { name: "Heterogeneous GNN", desc: "Separate node/edge encoders for spacetime vs field nodes — no weight sharing across types" },
      { name: "Message Passing", desc: "3-stage: field→ST aggregation, ST→ST propagation (1 hop = 1 ∂_μ), ST→field update" },
      { name: "Energy Head", desc: "Global readout computing S_E[φ] = Σ_x a² ℒ(x)" },
      { name: "Loss Function", desc: "Train as energy-based model: L = 𝔼[S_E[φ]] + KL-divergence from MC samples" }
    ],
    training: [
      "Generate ground truth configs via Metropolis-Hastings Monte Carlo on the same lattice",
      "Train GNN to score configurations: e^{-S_E} as unnormalized probability",
      "Optionally train normalizing flow on top for direct sampling",
      "Target observables: ⟨φ²⟩, two-point function G(x-y), critical exponent ν"
    ],
    milestones: [
      "Graph data pipeline: lattice → heterogeneous graph (PyG / DGL)",
      "GNN architecture implementing 3-stage message passing",
      "MC baseline dataset for 8×8, 16×16, 32×32 lattices",
      "Energy function recovery: GNN-scored configs match MC distribution",
      "Critical exponent measurement matching Ising class (ν ≈ 1)"
    ],
    risks: [
      { item: "Sign problem (Euclidean has none — this is why we start here)", severity: "low" },
      { item: "Over-smoothing in deep GNNs suppressing long-range correlations", severity: "medium" },
      { item: "Graph size scaling: 32⁴ in 4D is ~1M nodes", severity: "medium" }
    ],
    priorArt: {
      paper: "Bachtis, Aarts & Lucini — arXiv:2102.09449 · PRD 103, 074510 (2021)",
      title: "Quantum field-theoretic machine learning",
      overlap: "Uses a bipartite graph with φ⁴ scalar field theory on a square lattice, derived via the Hammersley-Clifford theorem. Nodes split into visible field nodes {φ₁…φₙ} and latent hidden units {h₁…hₘ}. Closest published work to our architecture.",
      distinction: "Their bipartition is visible matter ↔ latent units (RBM-style). Ours is spacetime geometry ↔ field content — a physically motivated split absent from their work. Direction also inverts: they run QFT→ML to derive a learning algorithm; we run ML→QFT to model field dynamics.",
      action: "Cite as Reference 1. Sharpen novelty: our bipartition is geometrically and physically motivated in a way theirs is not, and neither their framework nor homogeneous L-GCNs subsume ours."
    }
  },
  {
    id: 2,
    label: "Phase II",
    title: "U(1) Gauge Theory + Fermions",
    subtitle: "Euclidean QED in 2D/3D",
    status: "Core Extension",
    color: "#a78bfa",
    duration: "~4 months",
    goal: "Extend to gauge fields living on edges and integrate fermionic matter, establishing gauge-equivariant message passing as a first-class architectural primitive.",
    theory: {
      lagrangian: "ℒ_E = ψ̄(∂̸ + igA̸ + m)ψ + ¼F_μνF^μν",
      signature: "Euclidean",
      dimension: "2D Schwinger model, optionally 3D",
      rationale: "The Schwinger model (QED₂) is exactly solvable — mass gap, confinement, anomalies all known analytically. Simplest gauge theory with fermions."
    },
    graph: {
      spacetime_nodes: "Sites as before, features: [x^μ, a]",
      field_nodes: "ψ(x) nodes: features [spinor components ∈ ℂ²], φ if matter scalars added",
      st_st_edges: "Now carry gauge link U_μ(x) = e^{iaA_μ(x)} ∈ U(1) as edge feature",
      st_field_edges: "Covariant inhabits: ψ(x+μ̂) received via U_μ(x)·ψ(x) transformation"
    },
    architecture: [
      { name: "Gauge-Equivariant Layers", desc: "Message passing respects U(1) gauge symmetry: h_ψ(x) → e^{iα(x)}h_ψ(x), U_μ → e^{iα(x)}U_μe^{-iα(x+μ̂)}" },
      { name: "Plaquette Features", desc: "Wilson loops P_μν = U_μ(x)U_ν(x+μ̂)U†_μ(x+ν̂)U†_ν(x) computed as 4-cycle graph features — these are gauge-invariant" },
      { name: "Fermion Determinant", desc: "Integrate out ψ analytically → det(D̸+m) contributes as a reweighting factor on the gauge graph" },
      { name: "Complex-valued GNN", desc: "All features ∈ ℂ to handle spinor and phase structure" }
    ],
    training: [
      "Hybrid Monte Carlo (HMC) for gauge field configs — standard lattice QCD tooling",
      "Fermion determinant computed via sparse linear algebra (CG solver)",
      "GNN learns to approximate det(D̸+m) as a local graph computation (huge speedup if successful)",
      "Target: Wilson loop expectation values, string tension, mass gap"
    ],
    milestones: [
      "Gauge link edge features + plaquette cycle computation in graph",
      "Gauge-equivariant message passing layer (verify equivariance by construction)",
      "Fermion determinant approximation network",
      "String tension σ and mass gap m recovery matching analytic Schwinger result",
      "Generalization: train on one coupling β, evaluate on others"
    ],
    risks: [
      { item: "Fermion sign problem in finite-density — defer to Phase IV", severity: "low" },
      { item: "Gauge-equivariance hard to maintain through deep networks", severity: "high" },
      { item: "det(D̸+m) approximation may require very expressive networks", severity: "medium" }
    ]
  },
  {
    id: 3,
    label: "Phase III",
    title: "SU(3) Yang-Mills + Full QCD-like System",
    subtitle: "Euclidean 4D",
    status: "Full Realization",
    color: "#34d399",
    duration: "~6 months",
    goal: "Scale to SU(3) gauge group in 4D Euclidean spacetime — the graph architecture of lattice QCD — and benchmark against established lattice QCD results.",
    theory: {
      lagrangian: "ℒ_E = ½Tr[F_μνF^μν] + Σ_f ψ̄_f(D̸ + m_f)ψ_f",
      signature: "Euclidean 4D",
      dimension: "8⁴ to 16⁴ lattice (standard lattice QCD sizing)",
      rationale: "SU(3) is the real target. Success here means the architecture is competitive with state-of-the-art lattice QCD ML approaches."
    },
    graph: {
      spacetime_nodes: "N⁴ sites (4096 to 65536), features: [x^μ, a]",
      field_nodes: "6 quark flavor nodes per site + gluon field nodes, SU(3) spinor-color features",
      st_st_edges: "8 directed links per site, each carrying U_μ(x) ∈ SU(3) (3×3 complex matrix)",
      st_field_edges: "Covariant parallel transport in fundamental/adjoint representations"
    },
    architecture: [
      { name: "SU(3)-Equivariant Layers", desc: "Generalize U(1) equivariance to non-Abelian SU(3): non-commutative matrix edge features, ordered products along paths" },
      { name: "Multi-Scale Hierarchy", desc: "Coarse + fine spacetime subgraphs connected by pooling layers — this is RG flow implemented in the architecture" },
      { name: "Plaquette + Polyakov Loop Features", desc: "All gauge-invariant observables as pre-computed graph cycles" },
      { name: "Equivariant Attention", desc: "Attention over spacetime neighborhoods weighted by gauge-invariant distance measures" }
    ],
    training: [
      "Use existing lattice QCD configs (MILC, CLS ensembles publicly available)",
      "Train flow-based sampler: GNN generates gauge field proposals accepted/rejected by HMC",
      "Speedup metric: reduction in autocorrelation time vs pure HMC",
      "Target observables: hadron spectrum (pion, rho, nucleon masses), topological susceptibility"
    ],
    milestones: [
      "SU(3) matrix edge features + non-Abelian parallel transport in message passing",
      "Multi-scale spacetime graph (coarse-graining = RG step)",
      "Load and train on public MILC gauge configurations",
      "Pion mass and string tension recovery within 5% of benchmark",
      "Sampling autocorrelation improvement over pure HMC"
    ],
    risks: [
      { item: "Graph scale: 16⁴ × 8 links × SU(3) matrices = massive memory footprint", severity: "high" },
      { item: "Non-Abelian equivariance much harder — ordered products don't commute", severity: "high" },
      { item: "Topological freezing at fine lattice spacing — known hard problem", severity: "medium" }
    ]
  },
  {
    id: 4,
    label: "Phase IV",
    title: "Wick Rotation Bridge",
    subtitle: "Euclidean → Minkowski",
    status: "Analytic Continuation",
    color: "#fb923c",
    duration: "~4 months",
    goal: "Develop the analytic continuation pathway from Euclidean GNN representations to Minkowski real-time dynamics, using learned spectral functions as the bridge.",
    theory: {
      lagrangian: "G_E(τ) = ∫₀^∞ dω ρ(ω) e^{-ωτ}  [spectral representation]",
      signature: "Euclidean → Minkowski via τ = it",
      dimension: "2D first, then 4D",
      rationale: "Direct simulation in Minkowski is blocked by the sign problem. The standard route is: compute Euclidean correlators → extract spectral function ρ(ω) → Wick-rotate to get real-time Green's functions."
    },
    graph: {
      spacetime_nodes: "Add imaginary-time axis τ ∈ [0, β] (finite temperature formalism)",
      field_nodes: "Same as Euclidean phases",
      st_st_edges: "Temporal edges carry β (inverse temperature) as additional feature",
      st_field_edges: "Periodic BC for bosons, anti-periodic for fermions along τ"
    },
    architecture: [
      { name: "Spectral GNN Head", desc: "Output Euclidean correlator G_E(τ) from graph → learn spectral function ρ(ω) via inverse Laplace (ill-posed — regularize with neural network prior)" },
      { name: "Backus-Gilbert / MEM Layer", desc: "Maximum Entropy Method implemented as a differentiable layer — standard analytic continuation with learned entropy prior" },
      { name: "Padé Approximant Network", desc: "Learned rational function approximation for analytic continuation of 2-point functions in the complex plane" },
      { name: "Causal Masking", desc: "Enforce causality of retarded Green's function G^R(ω) = ∫ G_E with correct iε prescription" }
    ],
    training: [
      "Train on theories with known Minkowski spectra (free field: ρ(ω) = δ(ω²-m²))",
      "Supervised: G_E(τ) → ρ(ω) with analytic ρ as ground truth",
      "Unsupervised: enforce Kramers-Kronig relations as a physics constraint loss",
      "Validate: extract pion quasiparticle peak from finite-T QCD correlators"
    ],
    milestones: [
      "Finite-temperature graph with anti-periodic fermion BCs",
      "Differentiable MEM layer for spectral reconstruction",
      "Free field test: recover δ(ω²-m²) from G_E with <5% peak position error",
      "Interacting φ⁴: recover width broadening near criticality",
      "Real-time 2-point function G^R(t) via Fourier transform of learned ρ(ω)"
    ],
    risks: [
      { item: "Inverse Laplace is exponentially ill-conditioned — spectral reconstruction is hard", severity: "high" },
      { item: "MEM prior dependence can bias spectral shape — validate with multiple priors", severity: "medium" },
      { item: "Real-time dynamics (thermalization, non-equilibrium) still inaccessible this way", severity: "high" }
    ]
  },
  {
    id: 5,
    label: "Phase V",
    title: "Native Minkowski Formulation",
    subtitle: "Complex Langevin + Tensor Networks",
    status: "Frontier",
    color: "#f472b6",
    duration: "~6 months",
    goal: "Attempt direct real-time Minkowski path integral via complex Langevin dynamics and tensor network hybridization — attacking the sign problem head-on.",
    theory: {
      lagrangian: "ℒ_M = ½(∂_μφ)² - ½m²φ² - λφ⁴   [note: kinetic term now has SO(3,1) signature]",
      signature: "Minkowski (- + + +) or (+ - - -)",
      dimension: "1+1D first, targeting 3+1D",
      rationale: "Complex Langevin extends stochastic quantization to complex actions. Recent work shows it can evade the sign problem in some regimes. Tensor networks (MPS/MERA) handle 1+1D exactly."
    },
    graph: {
      spacetime_nodes: "Separate space and time dimensions explicitly; causal light-cone edge structure",
      field_nodes: "Complex-valued field nodes φ ∈ ℂ (Langevin complexification)",
      st_st_edges: "Causal edges only (past light cone) for Lorentzian causal set structure",
      st_field_edges: "Same bipartite structure; now with Lorentzian metric signature in edge features"
    },
    architecture: [
      { name: "Complex Langevin GNN", desc: "Stochastic gradient flow on complexified field space: ∂_τφ(x) = -δS_M/δφ* + η, GNN learns the drift term" },
      { name: "Lorentz-Equivariant Layers", desc: "SO(3,1)-covariant message passing — non-compact group, requires careful implementation via Lorentz algebra generators" },
      { name: "Tensor Network Hybrid", desc: "For 1+1D: replace spacetime subgraph with MPS/MERA tensor network; use GNN for field content, TN for entanglement structure" },
      { name: "Thimble Sampling", desc: "Lefschetz thimble deformation of integration contour — GNN learns the thimble geometry" }
    ],
    training: [
      "Complex Langevin: validate on theories where sign problem is known solvable (heavy-dense QCD)",
      "Monitor correctness criterion: absence of boundary terms in complexified field space",
      "Tensor network: exact diagonalization benchmarks in 1+1D",
      "Lorentz equivariance: verify under explicit boost transformations"
    ],
    milestones: [
      "Complex-valued GNN with Cauchy-Riemann constraint enforcement",
      "Complex Langevin drift learning: free field real-time propagator recovery",
      "Causal light-cone graph structure replacing hypercubic lattice",
      "SO(3,1) equivariant layer (even approximate) without Wick rotation",
      "1+1D φ⁴ real-time 2-point function matching thimble Monte Carlo"
    ],
    risks: [
      { item: "Complex Langevin can converge to wrong results — correctness criterion must be monitored", severity: "high" },
      { item: "SO(3,1) equivariance architecturally unsolved in general — active research area", severity: "high" },
      { item: "Sign problem may be fundamentally intractable for dense QCD — known NP-hard", severity: "high" }
    ]
  }
];

const literature = [
  { ref: "★ NEW", cite: "Bachtis et al. 2021", note: "Bipartite φ⁴ ML — closest prior; different bipartition" },
  { ref: "R1", cite: "Kanwar et al. 2020 PRL", note: "Equivariant flows for lattice gauge theory" },
  { ref: "R2", cite: "Favoni et al. 2022 PRL", note: "L-GCN — closest architecture; homogeneous" },
  { ref: "R3", cite: "Boyda et al. 2021 PRD", note: "SU(N) gauge-equivariant sampling" },
  { ref: "R4", cite: "Lehner & Wettig 2023", note: "Equivariant preconditioners; parallel transport layers" },
  { ref: "R5", cite: "Abbott et al. 2022", note: "Schwinger model at criticality — Phase II benchmark" },
  { ref: "R6", cite: "Urban & Pawlowski 2020", note: "Neural spectral reconstruction (Euclidean→Minkowski)" },
  { ref: "R7", cite: "Shi, Wang, Zhou 2022", note: "Spectral inversion is fundamentally ill-posed" },
];

const techStack = [
  { cat: "Graph Framework", items: ["PyTorch Geometric (PyG)", "DGL as alternative"], note: "Heterogeneous graph support built-in" },
  { cat: "Baselines / Benchmarks", items: ["Lattice QCD: openQCD, Grid", "Monte Carlo: numpy-based Metropolis"], note: "Reuse existing datasets (MILC configs)" },
  { cat: "Equivariance", items: ["e3nn for SO(3)", "Custom SU(N) layers"], note: "Lorentz group requires custom work" },
  { cat: "Flow Models", items: ["Normalizing flows (RealNVP/MAF)", "Consistency models for sampling"], note: "For path integral sampling" },
  { cat: "Hardware", items: ["A100 (80GB) for 4D lattices", "Multi-GPU for Phase III+"], note: "Graph sizes explode in 4D" },
  { cat: "Tensor Networks", items: ["ITensor (Julia)", "quimb (Python)"], note: "Phase V hybrid only" },
];

export default function GNNQFTPlan() {
  const [active, setActive] = useState(0);
  const phase = phases[active];

  const severityColor = (s) => s === "high" ? "#f87171" : s === "medium" ? "#fbbf24" : "#4ade80";

  return (
    <div style={{
      fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
      background: "#050a14",
      color: "#c8d8e8",
      minHeight: "100vh",
      padding: "0",
      overflowX: "hidden"
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Space+Mono:wght@400;700&display=swap');
        ::-webkit-scrollbar { width: 4px; } 
        ::-webkit-scrollbar-track { background: #0a1628; }
        ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 2px; }
        .phase-btn:hover { opacity: 1 !important; transform: translateX(3px); }
        .phase-btn { transition: all 0.2s ease; }
        .card { transition: border-color 0.2s; }
        .card:hover { border-color: rgba(255,255,255,0.15) !important; }
      `}</style>

      {/* Header */}
      <div style={{
        borderBottom: "1px solid #0f2a4a",
        padding: "24px 32px",
        background: "linear-gradient(to right, #050a14, #081525)",
        display: "flex",
        alignItems: "flex-end",
        gap: "24px",
        flexWrap: "wrap"
      }}>
        <div>
          <div style={{ fontSize: "10px", letterSpacing: "4px", color: "#4a7fa5", marginBottom: "6px", textTransform: "uppercase" }}>
            Research Plan · GNN-QFT
          </div>
          <h1 style={{ margin: 0, fontSize: "22px", fontFamily: "'Space Mono', monospace", color: "#e8f4ff", fontWeight: 700, lineHeight: 1.2 }}>
            Heterogeneous GNN<br/>
            <span style={{ color: "#00d4ff" }}>Field Theory</span> Architecture
          </h1>
        </div>
        <div style={{ marginLeft: "auto", textAlign: "right" }}>
          <div style={{ fontSize: "10px", color: "#4a7fa5", letterSpacing: "2px" }}>TOTAL TIMELINE</div>
          <div style={{ fontSize: "20px", color: "#00d4ff", fontFamily: "'Space Mono', monospace" }}>~23 months</div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", minHeight: "calc(100vh - 85px)" }}>

        {/* Sidebar */}
        <div style={{ borderRight: "1px solid #0f2a4a", padding: "20px 0", background: "#04080f" }}>
          {phases.map((p, i) => (
            <button
              key={p.id}
              className="phase-btn"
              onClick={() => setActive(i)}
              style={{
                display: "block",
                width: "100%",
                background: active === i ? "linear-gradient(to right, #0a1e36, #081525)" : "transparent",
                border: "none",
                borderLeft: `2px solid ${active === i ? p.color : "transparent"}`,
                padding: "14px 20px",
                textAlign: "left",
                cursor: "pointer",
                opacity: active === i ? 1 : 0.55
              }}
            >
              <div style={{ fontSize: "9px", letterSpacing: "3px", color: p.color, marginBottom: "3px" }}>{p.label}</div>
              <div style={{ fontSize: "11px", color: "#c8d8e8", lineHeight: 1.4 }}>{p.title}</div>
              <div style={{ fontSize: "9px", color: "#4a7fa5", marginTop: "3px" }}>{p.duration}</div>
            </button>
          ))}

          <div style={{ borderTop: "1px solid #0f2a4a", margin: "16px 0", padding: "16px 20px" }}>
            <div style={{ fontSize: "9px", letterSpacing: "3px", color: "#4a7fa5", marginBottom: "10px" }}>KEY LITERATURE</div>
            {literature.map((l, i) => (
              <div key={i} style={{ marginBottom: "8px" }}>
                <div style={{ display: "flex", gap: "6px", alignItems: "baseline" }}>
                  <span style={{
                    fontSize: "8px", color: l.ref === "★ NEW" ? "#fb923c" : "#4a7fa5",
                    fontWeight: l.ref === "★ NEW" ? "bold" : "normal",
                    whiteSpace: "nowrap"
                  }}>{l.ref}</span>
                  <span style={{ fontSize: "9px", color: l.ref === "★ NEW" ? "#fb923c" : "#7ea8c8" }}>{l.cite}</span>
                </div>
                <div style={{ fontSize: "9px", color: "#3a5a7a", paddingLeft: "6px", lineHeight: 1.4 }}>{l.note}</div>
              </div>
            ))}
          </div>

          <div style={{ borderTop: "1px solid #0f2a4a", margin: "16px 0", padding: "16px 20px" }}>
            <div style={{ fontSize: "9px", letterSpacing: "3px", color: "#4a7fa5", marginBottom: "10px" }}>TECH STACK</div>
            {techStack.map((t, i) => (
              <div key={i} style={{ marginBottom: "8px" }}>
                <div style={{ fontSize: "9px", color: "#7ea8c8", letterSpacing: "1px" }}>{t.cat}</div>
                {t.items.map((item, j) => (
                  <div key={j} style={{ fontSize: "9px", color: "#4a7fa5", paddingLeft: "6px" }}>· {item}</div>
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div style={{ padding: "28px 32px", overflowY: "auto" }}>

          {/* Phase header */}
          <div style={{ marginBottom: "24px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "8px", flexWrap: "wrap" }}>
              <span style={{
                fontSize: "9px", letterSpacing: "3px", padding: "3px 10px",
                border: `1px solid ${phase.color}`, color: phase.color, borderRadius: "2px"
              }}>{phase.status}</span>
              <span style={{ fontSize: "9px", color: "#4a7fa5", letterSpacing: "2px" }}>{phase.duration}</span>
            </div>
            <h2 style={{ margin: "0 0 4px 0", fontSize: "20px", fontFamily: "'Space Mono', monospace", color: "#e8f4ff" }}>
              {phase.title}
            </h2>
            <div style={{ fontSize: "12px", color: phase.color }}>{phase.subtitle}</div>
            <p style={{ marginTop: "12px", fontSize: "12px", lineHeight: 1.7, color: "#8aaccc", maxWidth: "680px" }}>
              {phase.goal}
            </p>
          </div>

          {/* Theory */}
          <div style={{ marginBottom: "20px" }}>
            <SectionLabel label="Theory" color={phase.color} />
            <div className="card" style={{
              background: "#07111e",
              border: "1px solid #0f2a4a",
              borderRadius: "4px",
              padding: "16px",
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
              gap: "12px"
            }}>
              {Object.entries(phase.theory).map(([k, v]) => (
                <div key={k}>
                  <div style={{ fontSize: "9px", letterSpacing: "2px", color: "#4a7fa5", marginBottom: "4px", textTransform: "uppercase" }}>{k}</div>
                  <div style={{ fontSize: "11px", color: "#c8d8e8", lineHeight: 1.5 }}>{v}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Graph Structure */}
          <div style={{ marginBottom: "20px" }}>
            <SectionLabel label="Graph Structure" color={phase.color} />
            <div className="card" style={{
              background: "#07111e",
              border: "1px solid #0f2a4a",
              borderRadius: "4px",
              padding: "16px",
              display: "grid",
              gap: "10px"
            }}>
              {Object.entries(phase.graph).map(([k, v]) => (
                <div key={k} style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: "12px", alignItems: "start" }}>
                  <div style={{ fontSize: "9px", letterSpacing: "1px", color: phase.color, paddingTop: "2px" }}>
                    {k.replace(/_/g, " ").toUpperCase()}
                  </div>
                  <div style={{ fontSize: "11px", color: "#8aaccc", lineHeight: 1.5 }}>{v}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Architecture + Training side by side */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "20px" }}>
            <div>
              <SectionLabel label="Architecture" color={phase.color} />
              <div style={{ display: "grid", gap: "8px" }}>
                {phase.architecture.map((a, i) => (
                  <div className="card" key={i} style={{
                    background: "#07111e",
                    border: "1px solid #0f2a4a",
                    borderRadius: "4px",
                    padding: "12px"
                  }}>
                    <div style={{ fontSize: "10px", color: phase.color, marginBottom: "4px" }}>{a.name}</div>
                    <div style={{ fontSize: "11px", color: "#7a9ab8", lineHeight: 1.5 }}>{a.desc}</div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <SectionLabel label="Training Strategy" color={phase.color} />
              <div className="card" style={{
                background: "#07111e",
                border: "1px solid #0f2a4a",
                borderRadius: "4px",
                padding: "12px"
              }}>
                {phase.training.map((t, i) => (
                  <div key={i} style={{
                    fontSize: "11px",
                    color: "#7a9ab8",
                    lineHeight: 1.6,
                    paddingBottom: i < phase.training.length - 1 ? "8px" : 0,
                    borderBottom: i < phase.training.length - 1 ? "1px solid #0a1e36" : "none",
                    marginBottom: i < phase.training.length - 1 ? "8px" : 0
                  }}>
                    <span style={{ color: phase.color }}>→ </span>{t}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Milestones */}
          <div style={{ marginBottom: "20px" }}>
            <SectionLabel label="Milestones" color={phase.color} />
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "8px" }}>
              {phase.milestones.map((m, i) => (
                <div className="card" key={i} style={{
                  background: "#07111e",
                  border: "1px solid #0f2a4a",
                  borderRadius: "4px",
                  padding: "10px 12px",
                  display: "flex",
                  gap: "10px",
                  alignItems: "flex-start"
                }}>
                  <span style={{
                    fontSize: "9px",
                    color: phase.color,
                    background: `${phase.color}18`,
                    border: `1px solid ${phase.color}44`,
                    borderRadius: "2px",
                    padding: "1px 6px",
                    marginTop: "2px",
                    whiteSpace: "nowrap"
                  }}>{String(i + 1).padStart(2, "0")}</span>
                  <div style={{ fontSize: "11px", color: "#8aaccc", lineHeight: 1.5 }}>{m}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Risks */}
          <div style={{ marginBottom: phase.priorArt ? "20px" : 0 }}>
            <SectionLabel label="Risks & Mitigations" color={phase.color} />
            <div style={{ display: "grid", gap: "6px" }}>
              {phase.risks.map((r, i) => (
                <div className="card" key={i} style={{
                  background: "#07111e",
                  border: "1px solid #0f2a4a",
                  borderRadius: "4px",
                  padding: "10px 12px",
                  display: "flex",
                  alignItems: "center",
                  gap: "12px"
                }}>
                  <div style={{
                    width: "6px", height: "6px",
                    borderRadius: "50%",
                    background: severityColor(r.severity),
                    flexShrink: 0,
                    boxShadow: `0 0 6px ${severityColor(r.severity)}`
                  }} />
                  <div style={{ fontSize: "11px", color: "#8aaccc", lineHeight: 1.5, flex: 1 }}>{r.item}</div>
                  <div style={{ fontSize: "9px", color: severityColor(r.severity), letterSpacing: "2px", whiteSpace: "nowrap" }}>
                    {r.severity.toUpperCase()}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Prior Art — shown only for phases with a known close neighbor */}
          {phase.priorArt && (
            <div>
              <SectionLabel label="Closest Prior Art" color="#fb923c" />
              <div style={{
                background: "#0d1a10",
                border: "1px solid #fb923c44",
                borderRadius: "4px",
                padding: "14px 16px"
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
                  <span style={{ fontSize: "9px", color: "#fb923c", letterSpacing: "2px", border: "1px solid #fb923c44", padding: "2px 8px", borderRadius: "2px" }}>★ MUST READ</span>
                  <div style={{ fontSize: "11px", color: "#fb923c", fontWeight: "bold" }}>{phase.priorArt.title}</div>
                </div>
                <div style={{ fontSize: "10px", color: "#7a9a7a", marginBottom: "6px", fontStyle: "italic" }}>{phase.priorArt.paper}</div>
                <div style={{ display: "grid", gap: "8px" }}>
                  {[
                    { label: "Overlap", text: phase.priorArt.overlap, color: "#8aaccc" },
                    { label: "Our distinction", text: phase.priorArt.distinction, color: "#a8d4a8" },
                    { label: "Action", text: phase.priorArt.action, color: "#fb923c" },
                  ].map(({ label, text, color }) => (
                    <div key={label} style={{ display: "grid", gridTemplateColumns: "100px 1fr", gap: "8px" }}>
                      <div style={{ fontSize: "9px", color: "#4a7fa5", letterSpacing: "1px", textTransform: "uppercase", paddingTop: "2px" }}>{label}</div>
                      <div style={{ fontSize: "11px", color, lineHeight: 1.6 }}>{text}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}

function SectionLabel({ label, color }) {
  return (
    <div style={{
      fontSize: "9px",
      letterSpacing: "3px",
      color: color,
      textTransform: "uppercase",
      marginBottom: "8px",
      display: "flex",
      alignItems: "center",
      gap: "8px"
    }}>
      <div style={{ width: "16px", height: "1px", background: color }} />
      {label}
    </div>
  );
}
