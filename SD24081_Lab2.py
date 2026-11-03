from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 300
    chromosome_length: int = 80
    target_ones: int = 50
    generations: int = 50
    crossover_rate: float = 0.90
    mutation_rate: float = 0.015
    tournament_size: int = 3
    elite_count: int = 2
    seed: int = 7

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError('Population size must be at least 2.')
        if self.chromosome_length < 2:
            raise ValueError('Chromosome length must be at least 2.')
        if not 0 <= self.target_ones <= self.chromosome_length:
            raise ValueError('Target number of ones must be within chromosome length.')
        if self.generations < 1:
            raise ValueError('Generations must be at least 1.')
        if not (0 <= self.crossover_rate <= 1 and 0 <= self.mutation_rate <= 1):
            raise ValueError('Rates must lie between 0 and 1.')
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError('Tournament size must be between 1 and population size.')
        if not 0 <= self.elite_count < self.population_size:
            raise ValueError('Elites must be fewer than population size.')


def fitness(population: np.ndarray, target_ones: int = 50, max_score: int = 80) -> np.ndarray:
    """Fitness for each row: max_score - abs(number_of_ones - target_ones)."""
    count_ones = population.sum(axis=1)
    return max_score - np.abs(count_ones - target_ones)


def tournament_selection(scores: np.ndarray, rng: np.random.Generator, k: int) -> int:
    candidates = rng.integers(0, len(scores), size=k)
    return int(candidates[np.argmax(scores[candidates])])


def crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One-point crossover. Does not alter parents."""
    cut = int(rng.integers(1, parent_a.size))
    return (np.concatenate((parent_a[:cut], parent_b[cut:])),
            np.concatenate((parent_b[:cut], parent_a[cut:])))


def mutate(child: np.ndarray, rng: np.random.Generator, rate: float) -> np.ndarray:
    """Independent bit-flip mutation for each gene."""
    result = child.copy()
    flags = rng.random(result.size) < rate
    result[flags] = 1 - result[flags]
    return result


def run_ga(config: GAConfig, on_progress: Callable[[int, int], None] | None = None) -> dict:
    """Run exactly `generations` reproduction cycles; log 0 and final generation."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    pop = rng.integers(0, 2, size=(config.population_size, config.chromosome_length), dtype=np.int8)
    history: list[dict] = []
    best_ever = None
    best_score = -np.inf
    first_hit = None

    for generation in range(config.generations + 1):
        scores = fitness(pop, config.target_ones, config.chromosome_length)
        best_index = int(np.argmax(scores))
        current_best = int(scores[best_index])
        if current_best > best_score:
            best_score = current_best
            best_ever = pop[best_index].copy()
        if first_hit is None and best_score == config.chromosome_length:
            first_hit = generation
        history.append({
            'Generation': generation,
            'Best fitness': current_best,
            'Average fitness': round(float(np.mean(scores)), 4),
            'Worst fitness': int(np.min(scores)),
            'Best-ever fitness': int(best_score),
            'Target matches': int(np.count_nonzero(scores == config.chromosome_length)),
        })
        if on_progress is not None:
            on_progress(generation, config.generations)
        if generation == config.generations:
            break

        elite_indices = np.argsort(scores)[-config.elite_count:] if config.elite_count else np.array([], dtype=int)
        next_population = [pop[i].copy() for i in elite_indices]
        while len(next_population) < config.population_size:
            i = tournament_selection(scores, rng, config.tournament_size)
            j = tournament_selection(scores, rng, config.tournament_size)
            a, b = pop[i], pop[j]
            if rng.random() < config.crossover_rate:
                c1, c2 = crossover(a, b, rng)
            else:
                c1, c2 = a.copy(), b.copy()
            for child in (c1, c2):
                if len(next_population) < config.population_size:
                    next_population.append(mutate(child, rng, config.mutation_rate))
        pop = np.stack(next_population)

    final_scores = fitness(pop, config.target_ones, config.chromosome_length)
    assert best_ever is not None
    return {
        'config': config,
        'best': best_ever,
        'best_fitness': int(best_score),
        'ones_count': int(best_ever.sum()),
        'zeros_count': int(config.chromosome_length - best_ever.sum()),
        'first_optimum_generation': first_hit,
        'history': pd.DataFrame(history),
        'population': pop.copy(),
        'scores': final_scores.copy(),
    }


# Streamlit interface - keep computation pure and UI separate.
def main() -> None:
    st.set_page_config(page_title='Genetic Algorithm | Lab 2', page_icon='🧬', layout='wide')
    st.title('🧬 Genetic Algorithm: 50 Ones in 80 Bits')
    st.write('**BSD3513 - Lab Report 2 | Chapter 2: Search Algorithms**')
    st.info('Assignment settings: population = 300, chromosome length = 80, target ones = 50, generations = 50, maximum fitness = 80.')

    with st.expander('Understanding the fitness function and GA steps', expanded=True):
        st.latex(r'F(x) = 80 - \left|\sum_{i=1}^{80}x_i - 50\right|')
        st.write('The closer the chromosome is to exactly 50 ones, the higher the fitness. At 50 ones the fitness is 80. More than 50 ones reduces fitness again.')
        st.markdown('**Workflow:** random initial population → fitness evaluation → tournament selection → one-point crossover → bit-flip mutation → elitism → next generation.')
        st.caption('Generation 0 is the initial population; generation 50 is the final result after 50 reproduction cycles.')

    with st.sidebar:
        st.header('Required assignment settings')
        st.metric('Population', 300)
        st.metric('Chromosome length', 80)
        st.metric('Target ones', 50)
        st.metric('Generations', 50)
        st.divider()
        st.header('Optional GA tuning')
        cross = st.slider('Crossover probability', 0.0, 1.0, 0.90, 0.05)
        mutation = st.slider('Mutation probability per bit', 0.0, 0.20, 0.015, 0.005)
        tournament = st.slider('Tournament size', 2, 10, 3)
        elite = st.slider('Elite chromosomes', 0, 10, 2)
        seed = st.number_input('Random seed', min_value=0, max_value=2**32 - 1, value=7, step=1)
        st.caption('Keep the seed fixed to reproduce identical results.')

    config = GAConfig(crossover_rate=cross, mutation_rate=mutation,
                      tournament_size=tournament, elite_count=elite, seed=int(seed))
    if st.button('▶ Run Genetic Algorithm', type='primary'):
        progress = st.progress(0)
        with st.spinner('Evaluating 50 generations...'):
            output = run_ga(config, on_progress=lambda gen, total: progress.progress(int(100 * gen / total)))
        st.session_state['ga_result'] = output
        st.session_state['ga_config'] = config
    result = st.session_state.get('ga_result')
    if result is None:
        st.write('Press **Run Genetic Algorithm** to generate a solution and view results.')
        return
    active_config = st.session_state['ga_config']
    if active_config != config:
        st.warning('You changed the tuning settings. Press Run Genetic Algorithm again to apply them.')

    is_optimal = result['best_fitness'] == 80
    st.success('Optimal 50-ones chromosome found!' if is_optimal else 'Run completed; an exact optimum was not found. Try a different seed or tuning parameters.')
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Best fitness / 80', result['best_fitness'])
    m2.metric('Ones / 50', result['ones_count'])
    m3.metric('Zeros / 30', result['zeros_count'])
    m4.metric('First optimum generation', str(result['first_optimum_generation']) if is_optimal else 'Not found')

    st.subheader('Best chromosome (80 bits)')
    chromosome = ''.join(str(int(bit)) for bit in result['best'])
    st.code(chromosome, language='text')
    st.caption(f'Length = {len(chromosome)} | Ones = {result["ones_count"]} | Zeros = {result["zeros_count"]}')
    st.subheader('Fitness over generations')
    st.line_chart(result['history'].set_index('Generation')[['Best fitness', 'Average fitness', 'Worst fitness']])
    st.caption('The best line cannot decrease when elitism > 0. Average and worst can change because variation is retained.')
    st.subheader('Generation-by-generation data')
    st.dataframe(result['history'], use_container_width=True, hide_index=True)
    st.download_button('Download fitness history CSV', data=result['history'].to_csv(index=False), file_name='GA_fitness_history.csv', mime='text/csv')
    with st.expander('Final population (all 300 chromosomes)'):
        details = pd.DataFrame({'Chromosome': [''.join(map(str, row.tolist())) for row in result['population']],
                                'Ones': result['population'].sum(axis=1).astype(int),
                                'Fitness': result['scores'].astype(int)})
        st.dataframe(details, use_container_width=True, hide_index=True)
        st.download_button('Download final population CSV', details.to_csv(index=False), file_name='GA_final_population.csv', mime='text/csv')
    st.markdown('**Reproducibility:** Same algorithm settings and random seed produce the same results. A GA is stochastic, so a different seed can produce a different path or result.')


if __name__ == '__main__':
    main()
