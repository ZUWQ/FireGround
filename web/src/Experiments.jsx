import { Fragment } from 'react'
import { symbolicResults, taskMetrics, teamResults, visualResults } from './experimentResults.js'
import teamFigure from './picture/team-organization.webp'

const metrics = {
  SVR: ['Saved Value Ratio', '%', '↑'],
  FCT: ['Fire Clearance Time', 's', '↓'],
  HPR: ['Health Protection Ratio', '%', '↑'],
  CRR: ['Civilian Rescue Rate', '%', '↑'],
  EC: ['Exploration Coverage', '%', '↑'],
  TG: ['Team Gain', 'pp', '↑'],
}

function MetricLabel({ code }) {
  const [name, unit, direction] = metrics[code]
  return <><abbr title={name}>{code}</abbr> {direction} <small>({unit})</small></>
}

function ScoreTable({ caption, firstColumn, columns, rows }) {
  return (
    <div className="experiment-table-scroll" role="region" aria-label={caption} tabIndex={0}>
      <table className="experiment-table">
        <caption>{caption}</caption>
        <thead>
          <tr>
            <th scope="col">{firstColumn}</th>
            {columns.map((code) => <th scope="col" key={code}><MetricLabel code={code} /></th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, ...values]) => (
            <tr key={label} data-row={label}>
              <th scope="row">{label}</th>
              {values.map((value, i) => <td key={columns[i]}>{value}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function VisualTable({ task }) {
  const columns = taskMetrics[task]
  const { models, human } = visualResults[task]
  return (
    <>
      <div className="experiment-table-scroll" role="region" aria-label={`Table 3, ${task} visual model results`} tabIndex={0}>
        <table className="experiment-table visual-scores">
          <caption>Table 3. {task} model results with RGB and RGB+DS</caption>
          <colgroup span={1} />
          {columns.map((code) => <colgroup key={code} span={2} />)}
          <thead>
            <tr>
              <th scope="col" rowSpan={2}>Model</th>
              {columns.map((code) => <th scope="colgroup" colSpan={2} key={code}><MetricLabel code={code} /></th>)}
            </tr>
            <tr>
              {columns.map((code) => (
                <Fragment key={code}><th scope="col">RGB</th><th scope="col">RGB+DS</th></Fragment>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map(({ model, rgb, rgbDs }) => (
              <tr key={model} data-row={model}>
                <th scope="row">{model}</th>
                {columns.map((code, i) => (
                  <Fragment key={code}><td>{rgb[i]}</td><td>{rgbDs[i]}</td></Fragment>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ScoreTable caption={`Table 3. ${task} descriptive human reference with RGB`}
        firstColumn="Operator" columns={columns} rows={[[task === 'T5' ? 'Human team' : 'Human', ...human]]} />
    </>
  )
}

function TaskPanel({ group, task, anchor, expanded, children }) {
  return (
    <details className="experiment-task" name={group} open={expanded}>
      <summary><strong>{task.id}</strong> {task.title}</summary>
      <div className="experiment-task-body" id={anchor} tabIndex={-1}>{children}</div>
    </details>
  )
}

export default function Experiments({ tasks }) {
  return (
    <section className="experiments band band--light" id="experiments" aria-labelledby="experiments-heading">
      <div className="section-heading">
        <h2 id="experiments-heading">Experiments</h2>
        <p>Results are averaged across six scenarios. See the <a href="#evaluation">evaluation protocol</a> for metric definitions and reference runs.</p>
      </div>

      <div className="experiment-groups">
        <section className="experiment-group" aria-labelledby="symbolic-results">
          <h3 id="symbolic-results">Symbolic Results</h3>
          <p>Five models use global symbolic observations on T1, T2 and T4, and local symbolic observations on T3 and T5.</p>
          {tasks.map((task) => (
            <TaskPanel key={task.id} group="symbolic-task" task={task} anchor={`symbolic-${task.id.toLowerCase()}`} expanded={task.id === 'T5'}>
              <ScoreTable caption={`Table 2. ${task.id} symbolic results`} firstColumn="Model"
                columns={taskMetrics[task.id]} rows={symbolicResults[task.id]} />
            </TaskPanel>
          ))}
        </section>

        <section className="experiment-group" aria-labelledby="visual-results">
          <h3 id="visual-results">Visual Results</h3>
          <p>Each pair of columns compares first-person RGB with the same visual interface augmented by object detections and segmentation masks (RGB+DS). Human participants used RGB.</p>
          {tasks.filter((task) => visualResults[task.id]).map((task) => (
            <TaskPanel key={task.id} group="visual-task" task={task}
              anchor={task.id === 'T5' ? 'visual-t5-deepseek' : 'visual-t3'} expanded={task.id === 'T5'}>
              <VisualTable task={task.id} />
            </TaskPanel>
          ))}
        </section>

        <section className="experiment-group" aria-labelledby="team-results">
          <h3 id="team-results">Team Organization</h3>
          <p>Table 4 holds the team at four firefighting robots and varies control and communication. Figure 3 varies team size and role composition.</p>
          {tasks.filter((task) => teamResults[task.id]).map((task) => (
            <TaskPanel key={task.id} group="team-task" task={task}
              anchor={task.id === 'T4' ? 'team-t4-deepseek' : 'team-t5'} expanded={task.id === 'T4'}>
              {teamResults[task.id].map(({ condition, rows }) => (
                <ScoreTable key={condition} caption={`Table 4. ${task.id} · ${condition}`} firstColumn="Control"
                  columns={['SVR', 'HPR', 'CRR', 'EC', 'TG']} rows={rows} />
              ))}
            </TaskPanel>
          ))}
          <figure className="paper-figure team-figure">
            <a href={teamFigure} aria-label="View Figure 3 at full resolution">
              <img src={teamFigure} alt="DeepSeek team-size and role-composition results. On T5, two firefighters and two rescue robots achieved the highest health protection under both controllers."
                width="1501" height="927" loading="lazy" decoding="async" />
            </a>
            <figcaption>
              <strong>Figure 3.</strong> DeepSeek with symbolic observations. Panel (a) compares one, two and four robots.
              Panel (b) varies the ratio of firefighting robots (FR) to rescue robots (RR) in four-robot T5 teams.
              Solid bars show decentralized control with communication. Hatched bars show centralized control.
              Health protection peaked with two robots of each role, while evacuation rates fell as the share of rescue robots increased.
            </figcaption>
          </figure>
        </section>

        <div className="supplementary-results">
          <a href={`${import.meta.env.BASE_URL}supplementary-results.pdf`}>Supplementary Results (PDF)</a>
          <p>Appendix H includes per-scenario results, dispersion statistics, coordination traces, decision-time experiments and fire-dynamics sweeps.</p>
        </div>
      </div>
    </section>
  )
}
