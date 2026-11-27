// 用 npm run dev 指令启动
import './App.css'
import overviewImageUrl from './picture/overview.webp'
import fireAgentsImageUrl from './picture/fireagents.webp'

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

const tasks = [
  {
    id: 'T1',
    title: 'Single-robot suppression',
    description: 'One firefighting robot uses global symbolic observations to choose which fire to suppress first.',
  },
  {
    id: 'T2',
    title: 'Add civilian rescue',
    description: 'One robot uses the same global view to suppress fires and escort civilians to safety. Civilian safety takes priority.',
  },
  {
    id: 'T3',
    title: 'Limit the view',
    description: 'One firefighting robot explores and suppresses fires using local symbolic or first-person visual observations.',
  },
  {
    id: 'T4',
    title: 'Add a robot team',
    description: 'Multiple firefighting robots use global symbolic observations to divide suppression work and reassign targets as the fire spreads.',
  },
  {
    id: 'T5',
    title: 'Combine exploration, rescue, and teamwork',
    description: 'A team uses local symbolic or visual observations to find fires and civilians, coordinate suppression, and escort civilians to safety.',
  },
]

const metricGroups = [
  {
    category: 'Fire suppression',
    metrics: [
      {
        name: 'Saved Value Ratio (SVR)',
        body: 'Percentage of property loss prevented relative to a matched free-burning run.',
      },
      {
        name: 'Fire Clearance Time (FCT)',
        body: 'Time until all fires are inactive. If fires remain at the time limit, the score adds 100 seconds for each remaining fire to the episode duration.',
      },
    ],
  },
  {
    category: 'Civilian rescue',
    metrics: [
      {
        name: 'Health Protection Ratio (HPR)',
        body: 'Percentage of civilian health loss prevented relative to a matched free-burning run.',
      },
      {
        name: 'Civilian Rescue Rate (CRR)',
        body: 'Percentage of civilians who are alive and in the safe zone at the end of the episode.',
      },
    ],
  },
  {
    category: 'Exploration and teamwork',
    metrics: [
      {
        name: 'Exploration Coverage (EC)',
        body: 'Percentage of eligible objects seen at least once by any robot during the episode.',
      },
      {
        name: 'Team Gain (TG)',
        body: 'Difference in SVR, in percentage points, between a team and a single robot with the same objectives and observation interface.',
      },
    ],
  },
]

const comparisonColumns = [
  { key: 'irreversibleLoss', title: 'Cumulative & Irreversible Loss' },
  { key: 'interventionCoupling', title: 'Intervention Coupling' },
  { key: 'inferenceProgression', title: 'Inference-Time Progression' },
  { key: 'exploration', title: 'Active Exploration' },
  { key: 'multiRobot', title: 'Multi-Robot Tasks' },
  { key: 'modality', title: 'Input Modality' },
  { key: 'tasks', title: 'Evaluated Tasks' },
]

const comparisonGroups = [
  {
    label: 'General embodied benchmarks',
    rows: [
      {
        benchmark: 'MAP-THOR',
        irreversibleLoss: 'Not reported',
        interventionCoupling: 'Not reported',
        inferenceProgression: 'Not reported',
        exploration: 'Yes',
        multiRobot: 'Yes',
        modality: 'Visual + Symbolic',
        tasks: 'Household tasks',
      },
      {
        benchmark: 'PARTNR',
        irreversibleLoss: 'Not reported',
        interventionCoupling: 'Not reported',
        inferenceProgression: 'Not reported',
        exploration: 'Yes',
        multiRobot: 'Yes',
        modality: 'Symbolic',
        tasks: 'Household tasks',
      },
    ],
  },
  {
    label: 'Dynamic and hazard-response benchmarks',
    rows: [
      {
        benchmark: 'FD-Bench',
        irreversibleLoss: 'Not reported',
        interventionCoupling: 'No',
        inferenceProgression: 'No',
        exploration: 'No',
        multiRobot: 'No',
        modality: 'Not reported',
        tasks: 'Escape planning',
      },
      {
        benchmark: 'HAZARD',
        irreversibleLoss: 'Yes',
        interventionCoupling: 'No',
        inferenceProgression: 'Not reported',
        exploration: 'Yes',
        multiRobot: 'No',
        modality: 'Symbolic',
        tasks: 'Object rescue',
      },
      {
        benchmark: 'CREW-Wildfire',
        irreversibleLoss: 'Yes',
        interventionCoupling: 'Yes',
        inferenceProgression: 'No',
        exploration: 'Yes',
        multiRobot: 'Yes',
        modality: 'Symbolic',
        tasks: 'Suppression, rescue',
      },
      {
        benchmark: 'FireNav',
        irreversibleLoss: 'Yes',
        interventionCoupling: 'No',
        inferenceProgression: 'Not reported',
        exploration: 'Yes',
        multiRobot: 'Yes',
        modality: 'Visual + Symbolic',
        tasks: 'Target search',
      },
      {
        benchmark: 'RescueBench',
        irreversibleLoss: 'Not reported',
        interventionCoupling: 'Not reported',
        inferenceProgression: 'Not reported',
        exploration: 'Yes',
        multiRobot: 'Yes',
        modality: 'Visual + Symbolic',
        tasks: 'Search and rescue',
      },
      {
        benchmark: 'RT-Safe',
        irreversibleLoss: 'Not reported',
        interventionCoupling: 'Yes',
        inferenceProgression: 'Yes',
        exploration: 'No',
        multiRobot: 'No',
        modality: 'Visual + Symbolic',
        tasks: 'Safe navigation',
      },
      {
        benchmark: 'ROAR',
        irreversibleLoss: 'Yes',
        interventionCoupling: 'Yes',
        inferenceProgression: 'Yes',
        exploration: 'Not reported',
        multiRobot: 'Yes',
        modality: 'Visual + Symbolic',
        tasks: 'Collection, combat',
      },
      {
        benchmark: 'FireGround',
        irreversibleLoss: 'Yes',
        interventionCoupling: 'Yes',
        inferenceProgression: 'Yes',
        exploration: 'Yes',
        multiRobot: 'Yes',
        modality: 'Visual + Symbolic',
        tasks: 'Suppression, rescue',
        ours: true,
      },
    ],
  },
]

function App() {
  return (
    <main id="top">
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

      <section className="hero">
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
          <h1>FireGround</h1>
          <p className="hero-copy">Benchmarking Embodied Decision-Making in Dynamic Residential Fires</p>
          <p className="hero-authors">Anonymous Authors</p>
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
          Dynamic embodied environments such as residential fire scenarios challenge agents to
          interpret evolving visual observations and act promptly as hazards evolve. We introduce
          FireGround, an Unreal Engine 5 benchmark coupling local visual perception, fire suppression,
          and civilian evacuation in 3D residential environments. Smoke and illumination changes
          affect observations, suppression alters fire progression, and civilian health deteriorates
          with exposure, requiring agents to balance information gathering with timely intervention.
          Its five tasks vary observation scope, rescue objectives, and robot teams, with metrics
          tailored to these challenges. We also introduce FireAgents, a standardized agent
          architecture supporting symbolic and visual observations under single-robot and centralized
          or decentralized multi-robot control. Evaluations of proprietary and open-source foundation
          models reveal lower performance under visual observations in several settings and
          difficulties translating broader exploration into better outcomes. Benefits from perception
          assistance and additional robots depend on task objectives and control structure. Together,
          FireGround and FireAgents support systematic evaluation of perception, reasoning, and action
          in dynamic embodied environments.
        </p>
      </section>

      <section className="overview band band--beige" id="overview">
        <div className="section-heading">
          <h2>Overview</h2>
        </div>
        <figure className="paper-figure">
          <a href={overviewImageUrl} aria-label="View Figure 1 at full resolution">
            <img
              src={overviewImageUrl}
              alt="Two response routes from the same initial fires, with the five FireGround tasks and their evaluation metrics."
              width="2400"
              height="1119"
              loading="lazy"
              decoding="async"
            />
          </a>
          <figcaption>
            <strong>Figure 1.</strong> Two rooms are on fire. Should the robots reach the study or
            the gym first? The two runs start with the same fires and show how that choice changes
            property damage and civilian health.
          </figcaption>
        </figure>

        <section className="task-configurations" id="tasks" aria-labelledby="tasks-heading">
          <h3 id="tasks-heading">Five task configurations</h3>
          <p>T2, T3, and T4 each change one feature of T1. T5 combines all three changes.</p>
          <dl className="task-list">
            {tasks.map((task) => (
              <div key={task.id}>
                <dt><span>{task.id}</span> {task.title}</dt>
                <dd>{task.description}</dd>
              </div>
            ))}
          </dl>
          <p className="task-note">
            The main T4 and T5 experiments use four firefighting robots. A separate T5 experiment
            varies the mix of firefighting and rescue robots.
          </p>
        </section>
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
        <figure className="paper-figure paper-figure--agents">
          <a href={fireAgentsImageUrl} aria-label="View Figure 2 at full resolution">
            <img
              src={fireAgentsImageUrl}
              alt="FireAgents connects visual or symbolic observations to decision-making, memory, and action execution. Urgent signals feed back into planning."
              width="1600"
              height="1153"
              loading="lazy"
              decoding="async"
            />
          </a>
          <figcaption>
            <strong>Figure 2.</strong> Each run uses one observation configuration chosen from Global Symbolic,
            Local Symbolic, RGB, and RGB+DS. RGB+DS adds object detections and segmentation masks.
            The execution system uses depth to map image targets into 3D. Civilian distress signals and
            urgent teammate messages trigger replanning.
          </figcaption>
        </figure>
        <p className="control-description">
          FireAgents supports single-robot control and two ways to organize a team. In decentralized
          control, each robot has its own decision-making agent. In centralized control, one agent
          assigns actions to the whole team. Robots execute their actions asynchronously in both.
        </p>
      </section>

      <section className="platform band band--dark" id="evaluation">
        <div className="section-heading">
          <h2>Evaluation</h2>
          <p className="evaluation-protocol">
            Each model–task condition is evaluated across six residential scenarios, with three runs
            per scenario. The applicable metrics depend on the <a href="#tasks">task configuration</a>.
          </p>
        </div>
        <div className="metric-grid">
          {metricGroups.map((group) => (
            <section className="metric-group" key={group.category}>
              <h3>{group.category}</h3>
              <dl className="metric-list">
                {group.metrics.map((metric) => (
                  <div className="metric-item" key={metric.name}>
                    <dt>{metric.name}</dt>
                    <dd>{metric.body}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
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
        <div className="related-work-table-wrap" role="region" aria-labelledby="simulator-comparison-heading" tabIndex={0}>
          <table className="related-work-table related-work-table--comparison">
            <caption className="related-work-caption">
              Table 1. Intervention coupling means robot actions can change how a hazard develops.
              Inference-time progression means the environment changes while a model decides.
              “Not reported” marks a capability the benchmark publication leaves unspecified.
            </caption>
            <colgroup span={1} />
            <colgroup span={2} />
            <colgroup span={5} />
            <thead>
              <tr className="related-work-table-head-group">
                <th rowSpan={2} scope="col">Benchmark</th>
                <th colSpan={2} scope="colgroup">Hazard Dynamics</th>
                <th colSpan={5} scope="colgroup">Evaluation Setup</th>
              </tr>
              <tr>
                {comparisonColumns.map((column) => (
                  <th key={column.key} scope="col">{column.title}</th>
                ))}
              </tr>
            </thead>
            {comparisonGroups.map((group) => (
              <tbody key={group.label}>
                <tr className="related-work-table-row--group">
                  <th scope="rowgroup" colSpan={comparisonColumns.length + 1}>{group.label}</th>
                </tr>
                {group.rows.map((row) => (
                  <tr key={row.benchmark} className={row.ours ? 'related-work-table-row--ours' : undefined}>
                    <th scope="row">{row.benchmark}</th>
                    {comparisonColumns.map((column) => (
                      <td key={column.key}>{row[column.key]}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            ))}
          </table>
        </div>
      </section>

      <section className="findings band band--beige" id="findings">
        <div className="section-heading">
          <h2>Findings</h2>
        </div>
        <div className="finding-result">
          <p>
            On T5 with local symbolic observations, GPT-6 and GPT-6.1 differed by just
            <strong> 1.01 percentage points</strong> in Saved Value Ratio. GPT-6.1’s Civilian Rescue Rate was
            <strong> 33.33 percentage points higher</strong>.
          </p>
          <table className="finding-table">
            <caption>
              Table 2, T5. Four firefighting robots under decentralized control with communication.
              Values are means across six scenarios, with three trials per scenario.
            </caption>
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Saved Value Ratio</th>
                <th scope="col">Civilian Rescue Rate</th>
              </tr>
            </thead>
            <tbody>
              <tr><th scope="row">GPT-6</th><td>40.98%</td><td>52.78%</td></tr>
              <tr><th scope="row">GPT-6.1</th><td>39.97%</td><td>86.11%</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </main>
  )
}

export default App
