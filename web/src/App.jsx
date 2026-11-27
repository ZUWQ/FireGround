// 用 npm run dev 指令启动
import { useEffect } from 'react'
import './App.css'
import Experiments from './Experiments.jsx'
import overviewImageUrl from './picture/overview.webp'
import fireAgentsImageUrl from './picture/fireagents.webp'

const HF_DEMO_VIDEO_BASE =
  'https://huggingface.co/datasets/ZWQ1103/temp_video/resolve/main'

const demonstrationVideoUrl = `${HF_DEMO_VIDEO_BASE}/FireGround.mp4`
const demonstrationVideoUrl2 = `${HF_DEMO_VIDEO_BASE}/real_environment_demo.mp4`
const demo1ImageUrl = `${HF_DEMO_VIDEO_BASE}/01DHP.webm`
const demo2ImageUrl = `${HF_DEMO_VIDEO_BASE}/02RCE.webm`
const demo3ImageUrl = `${HF_DEMO_VIDEO_BASE}/03IPD.webm`

const authors = [
  { name: 'Weiqin Zu', superscript: '1,2*' },
  { name: 'Yujie Lu', superscript: '2*' },
  { name: 'Miao Li', superscript: '2' },
  { name: 'Yifan Chen', superscript: '2,4' },
  { name: 'Huiying Li', superscript: '2' },
  { name: 'Yizhe Huang', superscript: '2,3' },
  { name: 'Kaiyang Cai', superscript: '2,5' },
  { name: 'Hualong Yan', superscript: '6' },
  { name: 'Ruihuan Zhang', superscript: '6' },
  { name: 'Dengji Zhao', superscript: '1†' },
  { name: 'Zhenliang Zhang', superscript: '2†' },
  { name: 'Xue Feng', superscript: '†' },
]

const affiliations = [
  { id: 1, name: 'ShanghaiTech University' },
  { id: 2, name: 'State Key Laboratory of General Artificial Intelligence, BIGAI' },
  { id: 3, name: 'Peking University' },
  { id: 4, name: 'Tsinghua University' },
  { id: 5, name: 'University of Science and Technology of China' },
  { id: 6, name: 'China Fire and Rescue Institute' },
]

const contactEmails = [
  'zuwq@shanghaitech.edu.cn',
  'zhaodj@shanghaitech.edu.cn',
  'zlzhang@bigai.ai',
  'feng.xue1580@gmail.com',
]

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

const overviewRuns = [
  {
    label: 'Policy A · Study first',
    file: 'first_fire.mp4',
    outcome: 'This route preserved more property.',
  },
  {
    label: 'Policy B · Gym first',
    file: 'first_civilian.mp4',
    outcome: 'Civilians lost less health on this route.',
  },
]

const keyFindings = [
  {
    title: 'Similar property savings came with different evacuation rates',
    description: 'GPT-6 and GPT-6.1 differed by 1.01 percentage points in Saved Value Ratio. GPT-6.1’s Civilian Rescue Rate was 33.33 percentage points higher.',
    resultAnchor: 'symbolic-t5',
    resultLink: 'View all T5 symbolic results',
    caption: 'Table 2. T5 with local symbolic observations and four firefighting robots under decentralized control with communication.',
    columns: ['Model', 'Saved Value Ratio', 'Civilian Rescue Rate'],
    rows: [
      ['GPT-6', '40.98%', '52.78%'],
      ['GPT-6.1', '39.97%', '86.11%'],
    ],
  },
  {
    title: 'The RGB team saw more objects and evacuated fewer civilians',
    description: 'With RGB alone, DeepSeek saw 81.39% of eligible objects and evacuated no civilians. With object detection and segmentation (RGB+DS), the team evacuated 16.67% of civilians.',
    resultAnchor: 'visual-t5-deepseek',
    resultLink: 'Compare T5 visual results',
    caption: 'Table 3. T5 with DeepSeek and four firefighting robots under decentralized control with communication.',
    columns: ['Visual input', 'Exploration Coverage', 'Civilian Rescue Rate'],
    rows: [
      ['RGB', '81.39%', '0%'],
      ['RGB+DS', '63.21%', '16.67%'],
    ],
  },
  {
    title: 'Four robots gained more under decentralized control',
    description: 'A single DeepSeek robot saved 32.51% of the property value that would have been lost in a free-burning run. Four centrally controlled robots saved less, while four communicating robots under decentralized control saved more.',
    resultAnchor: 'team-t4-deepseek',
    resultLink: 'Compare all T4 control settings',
    caption: 'Table 4. T4 with DeepSeek and global symbolic observations. Team Gain is the percentage-point difference from the matched single robot.',
    columns: ['Team control', 'Saved Value Ratio', 'Team Gain (pp)'],
    rows: [
      ['Centralized', '28.08%', '−4.43'],
      ['Decentralized with communication', '49.39%', '+16.88'],
    ],
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
        citation: { number: 21, url: 'https://openreview.net/forum?id=ZygZN5egzy', title: 'MAP-THOR: Benchmarking Long-Horizon Multi-Agent Planning Frameworks in Partially Observable Environments' },
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
        citation: { number: 3, url: 'https://arxiv.org/abs/2411.00081', title: 'PARTNR: A Benchmark for Planning and Reasoning in Embodied Multi-agent Tasks' },
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
        citation: { number: 44, url: 'https://escholarship.org/uc/item/4cd1t4p9', title: 'FD-Bench: Fine-Grained Evaluating the Decision-Making Capability of LLM Agents in Dynamic Scenarios' },
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
        citation: { number: 42, url: 'https://arxiv.org/abs/2401.12975', title: 'Hazard challenge: Embodied decision making in dynamically changing environments' },
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
        citation: { number: 10, url: 'https://arxiv.org/abs/2507.05178', title: 'CREW-Wildfire: Benchmarking agentic multi-agent collaborations at scale' },
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
        citation: { number: 1, url: 'https://firenav.github.io/', title: 'FireNav: A Physics-Grounded Benchmark for Cooperative Multi-Agent Navigation in Dynamic Indoor Fire Environments' },
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
        citation: { number: 39, url: 'https://arxiv.org/abs/2606.01848', title: 'RescueBench: Can Embodied Agents Save Lives in the Wild ?' },
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
        citation: { number: 40, url: 'https://arxiv.org/abs/2610.09294', title: 'RT-Safe: Benchmarking Agent Safety in Real-Time Embodied Environment' },
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
        citation: { number: 37, url: 'https://arxiv.org/abs/2610.24282', title: 'ROAR: Benchmarking LMM Agents under Real-Time Constraints in Multiplayer Action-Shooter Games' },
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
  useEffect(() => {
    function revealSection() {
      const target = document.getElementById(window.location.hash.slice(1))
      if (!target) return

      const disclosure = target.closest('details')
      if (disclosure) disclosure.open = true
      target.scrollIntoView()
    }

    // Fragment targets are mounted by React and may be inside a closed task panel.
    revealSection()
    window.addEventListener('hashchange', revealSection)
    return () => window.removeEventListener('hashchange', revealSection)
  }, [])

  return (
    <main id="top">
      <header className="site-header" aria-label="FireGround navigation">
        <a className="brand" href="#top" aria-label="FireGround home">
          FireGround
        </a>
        <nav>
          <a href="#abstract">Abstract</a>
          <a href="#overview">Overview</a>
          <a href="#findings">Key Findings</a>
          <a href="#simulation-preview">Simulation Preview</a>
          <a href="#challenge">Challenges</a>
          <a href="#comparison">Comparison</a>
          <a href="#fireagents">FireAgents</a>
          <a href="#evaluation">Evaluation Protocol</a>
          <a href="#experiments">Experiments</a>
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
          <div className="hero-byline">
            <ul className="hero-authors" aria-label="Authors" aria-describedby="author-notes">
              {authors.map(({ name, superscript }) => (
                <li key={name}>{name}<sup>{superscript}</sup></li>
              ))}
            </ul>

            <ul className="hero-affiliations" aria-label="Author affiliations">
              {affiliations.map(({ id, name }) => (
                <li key={id}><sup>{id}</sup> {name}</li>
              ))}
            </ul>

            <p className="author-notes" id="author-notes">
              <span><sup>*</sup> These authors contributed equally.</span>
              <span><sup>†</sup> Corresponding authors.</span>
            </p>
            <address className="author-contact" aria-label="Contact emails">
              {contactEmails.map((email) => (
                <a key={email} href={`mailto:${email}`}>{email}</a>
              ))}
            </address>
          </div>
          <div className="hero-actions" aria-label="Primary links">
            <a
              href="https://github.com/ZUWQ/FireGround"
              className="primary-link"
              target="_blank"
              rel="noopener noreferrer"
            >
              Code
            </a>
            <a
              href="https://huggingface.co/datasets/ZWQ1103/FireGround"
              className="secondary-link"
              target="_blank"
              rel="noopener noreferrer"
            >
              Dataset
            </a>
            <a
              href="https://drive.google.com/file/d/142bx8gi3KW1_9uYDwlMmb3jAImTkM21g/view?usp=drive_link"
              className="primary-link"
              target="_blank"
              rel="noopener noreferrer"
            >
              Simulator
            </a>
          </div>
          <details className="hero-abstract" lang="en">
            <summary>Full abstract</summary>
            <p id="abstract">
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
          </details>
        </div>
      </section>

      <section className="overview band band--beige" id="overview">
        <div className="section-heading">
          <h2>Overview</h2>
          <p className="overview-question">Two rooms are on fire. Should the robots reach the study or the gym first?</p>
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
            <strong>Figure 1.</strong> Both runs start with the same fires. The first room the robots
            reach changes how much property they preserve and how much harm civilians experience.
          </figcaption>
        </figure>

        <section className="overview-clips" aria-labelledby="overview-clips-heading">
          <h3 id="overview-clips-heading">Watch the two runs from Figure 1</h3>
          <div className="overview-clips-grid">
            {overviewRuns.map(({ label, file, outcome }) => (
              <figure key={file}>
                <figcaption>{label}</figcaption>
                <video
                  src={`${HF_DEMO_VIDEO_BASE}/${file}`}
                  aria-label={`Figure 1 run: ${label}`}
                  controls
                  playsInline
                  preload="metadata"
                />
                <p>{outcome}</p>
              </figure>
            ))}
          </div>
        </section>
      </section>

      <section className="findings band band--light" id="findings" aria-labelledby="findings-heading">
        <div className="section-heading">
          <h2 id="findings-heading">Key Findings</h2>
          <p>Values are means across six residential scenarios, with three trials per scenario for each condition.</p>
        </div>
        <div className="finding-list">
          {keyFindings.map(({ title, description, caption, columns, rows, resultAnchor, resultLink }) => (
            <article className="finding-result" key={title}>
              <div>
                <h3>{title}</h3>
                <p>{description}</p>
                <a href={`#${resultAnchor}`}>{resultLink}</a>
              </div>
              <table className="finding-table">
                <caption>{caption}</caption>
                <thead>
                  <tr>{columns.map((column) => <th key={column} scope="col">{column}</th>)}</tr>
                </thead>
                <tbody>
                  {rows.map(([condition, firstMetric, secondMetric]) => (
                    <tr key={condition}>
                      <th scope="row">{condition}</th>
                      <td>{firstMetric}</td>
                      <td>{secondMetric}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </article>
          ))}
        </div>
      </section>

      <section className="band band--beige" aria-labelledby="tasks-heading">
        <div className="task-configurations" id="tasks">
          <h2 id="tasks-heading">Five task configurations</h2>
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
          <details className="benchmark-comparison">
            <summary><h3 id="simulator-comparison-heading">Benchmark Comparison</h3></summary>
            <div id="comparison" className="related-work-table-wrap" role="region" aria-labelledby="simulator-comparison-heading" tabIndex={0}>
              <table className="related-work-table related-work-table--comparison">
                <caption className="related-work-caption">
                  Table 1. Intervention coupling means robot actions can change how a hazard develops.
                  Inference-time progression means the environment changes while a model decides.
                  “Not reported” marks an unspecified evaluation setting in the cited work.
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
                        <th scope="row">
                          {row.citation ? (
                            <a href={row.citation.url} title={row.citation.title}>
                              {row.benchmark} [{row.citation.number}]
                            </a>
                          ) : `${row.benchmark} (Ours)`}
                        </th>
                        {comparisonColumns.map((column) => (
                          <td key={column.key}>{row[column.key]}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                ))}
              </table>
            </div>
          </details>
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
          <h2>Evaluation Protocol</h2>
          <p className="evaluation-protocol">
            Each model condition covers six residential scenarios with three trials per scenario.
            After preburn, each trial runs for 300 seconds of environment time. Property and health
            protection use matched free-burning runs as references. Team Gain uses a matched single robot.
            In the main T4 and T5 comparisons, four firefighting robots use decentralized control with
            communication. Metrics vary by <a href="#tasks">task configuration</a>.
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

      <Experiments tasks={tasks} />
    </main>
  )
}

export default App
