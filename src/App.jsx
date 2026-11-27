// 用 npm run dev 指令启动
import './App.css'
import overviewImageUrl from './picture/overview.jpg?url'
import agentframeworkImageUrl from './picture/agent_framework.jpg?url'

const HF_DEMO_VIDEO_BASE =
  'https://huggingface.co/datasets/ZWQ1103/temp_video/resolve/main'

const demonstrationVideoUrl = `${HF_DEMO_VIDEO_BASE}/FireGround.mp4`
const demonstrationVideoUrl2 = `${HF_DEMO_VIDEO_BASE}/demo_third_vision.mov`
const demo1ImageUrl = `${HF_DEMO_VIDEO_BASE}/01DHP.webm`
const demo2ImageUrl = `${HF_DEMO_VIDEO_BASE}/02RCE.webm`
const demo3ImageUrl = `${HF_DEMO_VIDEO_BASE}/03IPD.webm`

const challenges = [
  {
    number: '01',
    title: 'Autonomous Hazard Progression',
    body: 'Residential fires spread while agents perceive, deliberate, and act, so target urgency and intervention feasibility keep changing without waiting for the team.',
  },
  {
    number: '02',
    title: 'Compounding and Irreversible Consequences',
    body: 'Object damage accumulates, smoke occludes egocentric perception, and civilian health declines over time; missed intervention windows may be permanently lost.',
  },
  {
    number: '03',
    title: 'Intervention-Coupled Dynamics',
    body: 'Suppression, protection, and evacuation reshape later fire propagation and task priorities, forcing agents to reason beyond each action\'s immediate effect.',
  },
]

const challengeDemos = [
  {
    id: 'demonstration-fire',
    src: demo1ImageUrl,
    ariaLabel: 'FireGround autonomous fire progression demonstration video',
  },
  {
    id: 'demonstration-extinguish',
    src: demo2ImageUrl,
    ariaLabel: 'FireGround compounding hazard escalation demonstration video',
  },
  {
    id: 'demonstration-rescue',
    src: demo3ImageUrl,
    ariaLabel: 'FireGround intervention-coupled rescue demonstration video',
  },
]

const taskCards = [
  {
    title: 'Fire Suppression',
    body: 'Three firefighting robots explore initially unknown rooms, locate burning objects, and suppress active fires before cascading damage expands.',
  },
  {
    title: 'Fire Suppression and Civilian Rescue',
    body: 'Three firefighting robots and one rescue robot must control fire, find civilians with degrading health, provide mask protection, and escort them to safer regions.',
  },
]

const metricGroups = [
  {
    category: 'Outcome Metrics',
    body: 'Outcome metrics evaluate how much damage the team prevents compared with a no-intervention baseline.',
    metrics: [
      {
        name: 'Saved Value Ratio (SVR)',
        body: 'Measures the proportion of environmental object value preserved after agent intervention.',
      },
      {
        name: 'Human Protection Ratio (HPR)',
        body: 'Measures the proportion of civilian health loss prevented through suppression, protection, and evacuation.',
      },
    ],
  },
  {
    category: 'Temporal Metric',
    body: 'The temporal metric captures whether agents suppress hazards quickly enough for their decisions to remain useful.',
    metrics: [
      {
        name: 'Completion Time (CT)',
        body: 'Tracks when all fires first become inactive, with a penalty for fires still active at the episode horizon.',
      },
    ],
  },
  {
    category: 'Diagnostic Metrics',
    body: 'Diagnostic metrics expose why a team succeeds or fails by measuring scaling, coverage, and consistency.',
    metrics: [
      {
        name: 'Team Gain (TG)',
        body: 'Measures the benefit of multi-agent operation over the corresponding single-agent baseline.',
      },
      {
        name: 'Exploration Coverage (EC)',
        body: 'Measures the fraction of environmental entities observed during an episode.',
      },
      {
        name: 'Performance Variability (PV)',
        body: 'Measures the range of SVR across repeated trials; lower values indicate more consistent behavior.',
      },
    ],
  },
]

const findings = [
  'Vision agents show a perception-action grounding bottleneck: high-level intent often fails to become effective navigation or suppression.',
  'Symbolic observations improve exploration and outcomes, but late intervention still keeps agent teams below human teams in cooperative fire suppression.',
  'Communication provides limited overall benefit in the reported settings, and decentralized asynchronous control is more responsive than centralized synchronized dispatch.',
  'Larger teams add capacity unevenly; redundant convergence, uneven task allocation, and crowding can offset the gains from extra robots.',
]

const comparisonColumns = [
  { key: 'benchmark', title: 'Benchmark', subtitle: '' },
  { key: 'autonomousHazard', title: 'Autonomous', subtitle: 'Hazard Progression' },
  { key: 'irreversible', title: 'Cascading &', subtitle: 'Irreversible' },
  { key: 'hazardAltering', title: 'Hazard-Altering', subtitle: 'Intervention' },
  { key: 'exploration', title: 'Egocentric 3D', subtitle: 'Exploration' },
  { key: 'multiAgent', title: 'Multi-Agent', subtitle: 'Decision-Making' },
  { key: 'visual', title: 'Visual', subtitle: 'Obs.' },
  { key: 'structured', title: 'Structured', subtitle: 'Obs.' },
]

const comparisonRows = [
  {
    type: 'group',
    label: 'General embodied multi-agent benchmarks',
  },
  {
    benchmark: 'MAP-THOR',
    autonomousHazard: 'no',
    irreversible: 'no',
    hazardAltering: 'no',
    exploration: 'yes',
    multiAgent: 'yes',
    visual: 'yes',
    structured: 'yes',
  },
  {
    benchmark: 'Habitat-MAS',
    autonomousHazard: 'no',
    irreversible: 'no',
    hazardAltering: 'no',
    exploration: 'no',
    multiAgent: 'yes',
    visual: 'no',
    structured: 'yes',
  },
  {
    benchmark: 'RoCoBench',
    autonomousHazard: 'no',
    irreversible: 'no',
    hazardAltering: 'no',
    exploration: 'no',
    multiAgent: 'yes',
    visual: 'no',
    structured: 'yes',
  },
  {
    benchmark: 'VIKI-Bench',
    autonomousHazard: 'no',
    irreversible: 'no',
    hazardAltering: 'no',
    exploration: 'partial',
    multiAgent: 'yes',
    visual: 'yes',
    structured: 'no',
  },
  {
    benchmark: 'Virtual Community',
    autonomousHazard: 'no',
    irreversible: 'no',
    hazardAltering: 'no',
    exploration: 'yes',
    multiAgent: 'yes',
    visual: 'yes',
    structured: 'yes',
  },
  {
    type: 'group',
    label: 'Hazard and disaster-response benchmarks',
  },
  {
    benchmark: 'RMASBench',
    autonomousHazard: 'yes',
    irreversible: 'yes',
    hazardAltering: 'yes',
    exploration: 'no',
    multiAgent: 'yes',
    visual: 'no',
    structured: 'yes',
  },
  {
    benchmark: 'FD-Bench',
    autonomousHazard: 'yes',
    irreversible: 'partial',
    hazardAltering: 'no',
    exploration: 'no',
    multiAgent: 'no',
    visual: 'no',
    structured: 'yes',
  },
  {
    benchmark: 'HAZARD',
    autonomousHazard: 'yes',
    irreversible: 'yes',
    hazardAltering: 'no',
    exploration: 'yes',
    multiAgent: 'no',
    visual: 'yes',
    structured: 'yes',
  },
  {
    benchmark: 'CREW-Wildfire',
    autonomousHazard: 'yes',
    irreversible: 'yes',
    hazardAltering: 'yes',
    exploration: 'no',
    multiAgent: 'yes',
    visual: 'yes',
    structured: 'yes',
  },
  {
    benchmark: 'VULCAN',
    autonomousHazard: 'yes',
    irreversible: 'partial',
    hazardAltering: 'no',
    exploration: 'yes',
    multiAgent: 'yes',
    visual: 'yes',
    structured: 'no',
  },
  {
    benchmark: 'RescueBench',
    autonomousHazard: 'no',
    irreversible: 'no',
    hazardAltering: 'no',
    exploration: 'yes',
    multiAgent: 'partial',
    visual: 'yes',
    structured: 'no',
  },
  {
    benchmark: 'ESARBench',
    autonomousHazard: 'no',
    irreversible: 'no',
    hazardAltering: 'no',
    exploration: 'yes',
    multiAgent: 'no',
    visual: 'yes',
    structured: 'no',
  },
  {
    benchmark: 'FireGround',
    autonomousHazard: 'yes',
    irreversible: 'yes',
    hazardAltering: 'yes',
    exploration: 'yes',
    multiAgent: 'yes',
    visual: 'yes',
    structured: 'yes',
    ours: true,
  },
]

const supportMark = (value) => {
  if (value === 'yes') return 'Yes'
  if (value === 'partial') return '~'
  return '-'
}


function App() {
  return (
    <main>
      <header className="site-header" aria-label="FireGround navigation">
        <a className="brand" href="#top" aria-label="FireGround home">
          FireGround
        </a>
        <nav>
          <a href="#abstract">Abstract</a>
          <a href="#overview">Overview</a>
          <a href="#simulation-preview">Simulation Preview</a>
          <a href="#challenge">Challenges</a>
          <a href="#fireagents">FireAgents</a>
          <a href="#evaluation">Evaluation</a>
          <a href="#comparison">Comparison</a>
          <a href="#findings">Findings</a>
        </nav>
      </header>

      <section className="hero" id="top">
        <video
          className="hero-video"
          src={demonstrationVideoUrl}
          autoPlay
          muted
          loop
          playsInline
          aria-hidden="true"
        />
        <div className="hero-shade" />
        <div className="hero-content">
          <p className="eyebrow">A Benchmark for Embodied Multi-Agent Decision-Making in Residential Fires</p>
          <h1>FireGround</h1>
          <p className="hero-copy">
          Anonymous Authors
          </p>
          <div className="hero-actions" aria-label="Primary links">
            <a
              href="https://anonymous.4open.science/r/FireGround/"
              className="primary-link"
              target="_blank"
              rel="noopener noreferrer"
            >
              Code
            </a>
            <a
              href="https://huggingface.co/datasets/EMAS4Rescue/FireGround"
              className="secondary-link"
              target="_blank"
              rel="noopener noreferrer"
            >
              Dataset
            </a>
            <a
              href="https://drive.google.com/file/d/1wgos_ZZG2T6XC8TU00DXmWDLy1ua4tHv/view?usp=drive_link"
              className="primary-link"
              target="_blank"
              rel="noopener noreferrer"
            >
              Simulator
            </a>
          </div>
        </div>
      </section>

      <section className="abstract band band--light" lang="en" id="abstract">
        <div className="section-heading">
          <h2>Abstract</h2>
        </div>
        <p className="lead abstract-lead">
        Disaster response is a canonical testbed for embodied multi-agent systems (EMAS), requiring embodied perception, reasoning about spreading hazards, prediction of others’ future behavior, and coordinated intervention under time pressure. Existing benchmarks, however, rarely capture the coupled dynamics. To close this gap, we introduce FireGround, a high-fidelity UE5 benchmark for residential fires, smoke occlusion, declining civilian health, and intervention-coupled dynamics. It provides egocentric visual and symbolic observations, enabling controlled evaluation of visual grounding and downstream decision-making. Heterogeneous firefighting and rescue agents perform fire suppression and joint fire-rescue tasks. FireGround supports automated evaluation of task outcomes, intervention timeliness, spatial exploration, and coordination across coordination structures and team sizes. We further introduce FireAgents, a model-agnostic agent framework that connects diverse foundation models to FireGround through a standardized perception-decision-action interface. Results show that visual agents struggle to ground their observations into effective actions. Symbolic observations improve exploration and task outcomes, but agents still intervene late and fall behind human teams. Communication yields limited gains, while larger teams are often undermined by redundant actions, uneven task allocation, and crowding that restricts movement and visibility. Current models therefore struggle to convert environmental information and added team capacity into timely, complementary interventions.
        </p>
      </section>

      <section className="overview band band--beige" id="overview">
        <div className="section-heading">
          <h2>Overview</h2>
        </div>
        <div className="figure-block">
          <div className="overview-image-wrap">
            <img
              src={overviewImageUrl}
              alt="FireGround overview figure"
              className="overview-image"
              loading="lazy"
              decoding="async"
            />
          </div>
          <div className="figure-placeholder">
            Figure 1: Overview of FireGround. A high-fidelity environment models autonomously propagating fire and provides egocentric visual or symbolic observations. Heterogeneous robots perform fire-suppression and joint fire-rescue tasks under decentralized asynchronous control with inter-agent communication; centralized task dispatch serves as a coordination ablation. Evaluation uses outcome, temporal, and diagnostic metrics.
          </div>
        </div>
      </section>

      <section className="simulation-preview band band--dark" id="simulation-preview">
        <div className="section-heading">
          <h2>Simulation&nbsp;Preview</h2>
        </div>
        <div className="video-frame video-frame--double">
          <video
            src={demonstrationVideoUrl}
            controls
            playsInline
            preload="metadata"
            aria-label="FireGround simulation preview overview video"
          />
          <video
            src={demonstrationVideoUrl2}
            controls
            playsInline
            preload="metadata"
            aria-label="FireGround simulation preview third-person video"
          />
        </div>
      </section>

      <section className="challenge band band--light" id="challenge">
        <div className="challenge-layout" id="small_demo">
          <div className="challenge-header-row">
            <div className="challenge-header-title">
              <h2>Challenges</h2>
            </div>
            <div className="challenge-header-demo-title">
              <p className="eyebrow">Property Demos</p>
            </div>
          </div>
          {challenges.map((challenge, i) => {
            const demo = challengeDemos[i]
            return (
              <div className="challenge-row" key={challenge.title}>
                <article className="track">
                  <span>{challenge.number}</span>
                  <div>
                    <h3>{challenge.title}</h3>
                    <p>{challenge.body}</p>
                  </div>
                </article>
                <div className="challenge-demo-block" id={demo.id}>
                  <div className="video-frame video-frame--compact">
                    <video
                      src={demo.src}
                      controls
                      playsInline
                      preload="metadata"
                      aria-label={demo.ariaLabel}
                    />
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </section>

      <section className="agent-interaction band band--beige" id="fireagents">
        <div className="section-heading">
          <h2>FireAgents</h2>
        </div>
        <div className="figure-block">
          <div className="overview-image-wrap">
            <img
              src={agentframeworkImageUrl}
              alt="FireGround FireAgents interaction figure"
              className="overview-image"
              loading="lazy"
              decoding="async"
            />
          </div>
          <div className="figure-placeholder">
            Figure 2: Overview of FireAgents. At each decision step, an agent maps local
            symbolic or RGB observations and context to a high-level action, which is
            grounded and executed in the environment. Depth supports visual action grounding
            without being exposed to the model, while execution feedback informs the next
            step.
          </div>
        </div>
      </section>

      <section className="platform band band--dark" id="evaluation">
        <div className="section-heading">
          <h2>Evaluation</h2>
        </div>
        <div className="evaluation-module">
          <div className="evaluation-module-heading">
            <h3>Tasks</h3>
          </div>
          <div className="principle-grid principle-grid--tasks">
            {taskCards.map((card) => (
              <article className="principle-card" key={card.title}>
                <h3>{card.title}</h3>
                <p>{card.body}</p>
              </article>
            ))}
          </div>
        </div>
        <div className="evaluation-module">
          <div className="evaluation-module-heading">
            <h3>Metrics</h3>
          </div>
          <div className="metric-grid">
            {metricGroups.map((group) => (
              <article className="metric-group" key={group.category}>
                <h3>{group.category}</h3>
                <p>{group.body}</p>
                <dl className="metric-list">
                  {group.metrics.map((metric) => (
                    <div className="metric-item" key={metric.name}>
                      <dt>{metric.name}</dt>
                      <dd>{metric.body}</dd>
                    </div>
                  ))}
                </dl>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section
        className="related-work band band--light"
        id="comparison"
        aria-labelledby="simulator-comparison-heading"
      >
        <div className="section-heading">
          <h2 id="simulator-comparison-heading">Comparison</h2>
        </div>
        <div className="related-work-table-wrap">
          <table className="related-work-table related-work-table--comparison">
            <caption className="related-work-caption">
              Comparison with representative embodied multi-agent and
              disaster-response benchmarks. Yes, ~, and - denote full, partial
              or restricted, and no support, respectively. Entries reflect
              evaluated settings rather than all capabilities of the underlying
              simulators. Visual inputs are image-based; structured inputs
              contain semantic, geometric, or state information.
            </caption>
            <thead>
              <tr className="related-work-table-head-group">
                <th scope="col" aria-label="Benchmark" />
                <th colSpan={3} scope="colgroup">
                  Hazard Dynamics
                </th>
                <th colSpan={4} scope="colgroup">
                  Embodied Evaluation
                </th>
              </tr>
              <tr>
                {comparisonColumns.map((column) => (
                  <th scope="col" key={column.key}>
                    <span className="related-work-th-l1">{column.title}</span>
                    <br />
                    <span className="related-work-th-l2">{column.subtitle || '\u00a0'}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {comparisonRows.map((item) => {
                if (item.type === 'group') {
                  return (
                    <tr className="related-work-table-row--group" key={item.label}>
                      <th colSpan={comparisonColumns.length} scope="rowgroup">
                        {item.label}
                      </th>
                    </tr>
                  )
                }

                return (
                  <tr
                    key={item.benchmark}
                    className={item.ours ? 'related-work-table-row--ours' : undefined}
                  >
                    {comparisonColumns.map((column) => (
                      <td key={column.key}>
                        {column.key === 'benchmark'
                          ? item.benchmark
                          : supportMark(item[column.key])}
                      </td>
                    ))}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="findings band band--beige" id="findings">
        <div className="section-heading">
          <h2>Findings</h2>
        </div>
        <ul className="finding-list">
          {findings.map((finding) => (
            <li key={finding}>{finding}</li>
          ))}
        </ul>
      </section>
    </main>
  )
}

export default App
