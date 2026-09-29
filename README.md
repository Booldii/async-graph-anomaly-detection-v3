# Continuous Risk Scoring for Ethereum Wallets
(Polish version below)

I develop this project as the capstone for Sages' *AI & Machine Learning Engineering* course, building a live
risk-scoring system that reacts to Ethereum wallet activity on the fly. Because I wanted to tackle a streaming
problem, the system acts as a live predictor that updates wallet risk profiles in real time after every transaction.

Ultimately, I built this repository as a showcase of MLOps and systems-level thinking, demonstrating how solid pipeline
design and experiment tracking can be applied to messy, streaming data.

## How it works, in plain terms

**Stage 1 - Offline labeling.** (finished) There's no way to get freshly labeled fraud data for current
Ethereum activity, so the project starts from a historical, hand-labeled dataset (the February
2025 ByBit hack and its money-laundering trail) and trains a classifier on it. That classifier is
then used to generate probabilistic "soft" risk labels for fresh, unlabeled transfers pulled live
from BigQuery - a deliberate, documented workaround for the lack of ground truth on current data.

**Stage 2 - Temporal graph training** *(planned, not yet started)*. A graph-based model that
tracks how an address's behavior evolves over time, using the soft labels from Stage 1 as its
training target.

**Stage 3 - Live inference** *(planned, not yet started)*. A stateful, event-driven service that
updates an address's risk state as new transactions arrive and raises an alert when its risk
score jumps sharply.

## Current status

**Built and working (Stage 1):**
- Exploratory analysis of the historical labeled dataset
- A pipeline that pulls fresh Ethereum token-transfer data from BigQuery, deduplicates it, flags
  smart-contract addresses, and draws an activity-stratified address sample
- A data-quality sanity check on that fresh pull (duplicate diagnosis, schema drift, population
  overlap with the historical dataset, feature-distribution comparison)
- A feature-engineering module that runs in two modes: building a labeled training set from the
  historical data, or scoring features for fresh, unlabeled addresses
- An ensemble of independently-trained classifiers (one per feature family) with experiment
  tracking, SHAP-based diagnostics, and an automated acceptance gate against any single feature
  family dominating the result
- Generating soft risk labels for the sampled fresh addresses using the trained ensemble.

**Not started yet:** Stage 2 (temporal graph model) and Stage 3 (live inference service).

## Repository layout

```
src/
  data/       scripts that pull or prepare data (BigQuery fetches, contract lookups, sampling)
  features/   turns raw transfer data into a per-address feature table
  models/     trains the classifier ensemble
notebooks/    exploratory analysis
data/         raw/ interim/ processed/
mlruns/       local MLflow experiment-tracking store
```

## A few design decisions worth knowing upfront

- **Scope is deliberately narrow**: only ERC-20 token transfer events, and only externally-owned accounts are scored.
  Both are documented simplifications, not gaps that were missed.
- **Transfer values are normalized without external price data**: a robust z-score (median/MAD of
  the log value), computed separately per token contract. This is self-referential - recomputed
  from whatever data is on hand rather than relying on a maintained price list - so it's resilient
  to token/market drift over time.
- **The classifier is an ensemble of feature-family specialists** (value, gas, behavioral,
  structural) rather than one model over all features. This avoids a single feature family
  dominating the result and gives a rough, secondary signal (how much the families disagree) for free.
- **Every script that can touch billed cloud resources defaults to a dry run.** Actually pulling
  data or querying BigQuery for real always requires an explicit "--execute" flag.

## Known limitations

- The historical training data comes from a single incident. Its label balance (~26% flagged as
  fraud-related) and address behavior don't reflect real-world Ethereum traffic, so soft labels
  are treated as a relative risk ranking, not a calibrated probability.
- The sanity-check notebook found a real distributional shift between the historical window and
  fresh data (eg. typical gas cost) - a reminder that the trained ensemble is extrapolating,
  rather than interpolating, when scoring current activity.
- Blending the ensemble's output with hand-written heuristics was part of the original Stage 1
  plan. for this pass - the ensemble's own consensus is used on its own and the heuristic blend
  was dropped as an unnecessary complication for now.

## Quickstart

```bash
uv sync
```

Pipeline scripts, in order (each of the BigQuery-touching ones prints a cost estimate and does
nothing else unless you pass "--execute"):

```bash
uv run python src/data/bq_fetch_day.py            # pull fresh transfer data
uv run python src/data/bq_contracts_fresh.py       # flag which fresh addresses are contracts
uv run python src/data/sample_fresh_addresses.py   # dedup + sample a subset of addresses
uv run python src/features/build_features.py --mode train   # historical, labeled feature table
uv run python src/features/build_features.py --mode score   # fresh, unlabeled feature table
uv run python src/models/train_xgboost_ensemble.py           # trains the ensemble, logs to MLflow
```

---

# Szacowanie ryzyka live dla portfeli Ethereum

Rozwijam ten projekt jako projekt końcowy kursu *AI & Machine Learning Engineering*, organizowanego przez Sages, budując system oceny ryzyka
działający na żywo, który na bieżąco reaguje na aktywność portfeli Ethereum. Ponieważ chciałem zmierzyć się z problemem
strumieniowym, system działa jako predyktor online, który aktualizuje profile ryzyka portfeli w czasie rzeczywistym po każdej transakcji.

Ostatecznie repozytorium to ma być prezentacją podejścia MLOps i myślenia systemowego - pokazuje, jak solidny projekt
pipeline'u i śledzenie eksperymentów można zastosować do nieuporządkowanych danych strumieniowych.

## Jak to działa, w prostych słowach

**Etap 1 - Etykietowanie offline.** (ukończony) Nie da się uzyskać świeżo oetykietowanych danych o oszustwach dla bieżącej
aktywności w sieci Ethereum, dlatego projekt wychodzi od historycznego, ręcznie oetykietowanego zbioru danych (włamanie
na ByBit z lutego 2025 r. i związany z nim ślad prania pieniędzy) i trenuje na nim klasyfikator. Klasyfikator ten jest
następnie używany do generowania probabilistycznych, „miękkich" etykiet ryzyka dla świeżych, nieoetykietowanych transferów
pobieranych na żywo z BigQuery - to świadome, udokumentowane obejście braku danych referencyjnych (ground truth) dla bieżących danych.

**Etap 2 - Trenowanie temporalnego modelu grafowego** *(planowany, jeszcze nierozpoczęty)*. Model grafowy, który
śledzi, jak zachowanie adresu zmienia się w czasie, wykorzystując miękkie etykiety z Etapu 1 jako cel treningowy.

**Etap 3 - Inferencja na żywo** *(planowany, jeszcze nierozpoczęty)*. Stanowa, sterowana zdarzeniami usługa, która
aktualizuje stan ryzyka adresu w miarę napływu nowych transakcji i generuje alert, gdy jego wskaźnik ryzyka
gwałtownie wzrośnie.

## Aktualny stan

**Zbudowane i działające (Etap 1):**
- Eksploracyjna analiza historycznego, oetykietowanego zbioru danych
- Pipeline, który pobiera świeże dane o transferach tokenów Ethereum z BigQuery, usuwa duplikaty, oznacza
  adresy smart kontraktów i losuje próbkę adresów warstwowaną według poziomu aktywności
- Kontrola jakości danych dla tego świeżego pobrania (diagnoza duplikatów, dryf schematu, pokrycie populacji
  z historycznym zbiorem danych, porównanie rozkładów cech)
- Moduł inżynierii cech działający w dwóch trybach: budowanie oetykietowanego zbioru treningowego z danych
  historycznych lub wyliczanie cech dla świeżych, nieoetykietowanych adresów
- Zespół (ensemble) niezależnie trenowanych klasyfikatorów (po jednym na każdą rodzinę cech) ze śledzeniem eksperymentów,
  diagnostyką opartą na SHAP i automatyczną bramką akceptacyjną chroniącą przed zdominowaniem wyniku przez
  jedną rodzinę cech
- Generowanie miękkich etykiet ryzyka dla wylosowanych świeżych adresów przy użyciu wytrenowanego zespołu.

**Jeszcze nierozpoczęte:** Etap 2 (temporalny model grafowy) i Etap 3 (usługa inferencji na żywo).

## Struktura repozytorium

```
src/
  data/       skrypty pobierające lub przygotowujące dane (zapytania do BigQuery, wyszukiwanie kontraktów, próbkowanie)
  features/   przekształca surowe dane o transferach w tabelę cech per adres
  models/     trenuje zespół klasyfikatorów
notebooks/    analiza eksploracyjna
data/         raw/ interim/ processed/
mlruns/       lokalne repozytorium śledzenia eksperymentów MLflow
```

## Kilka decyzji projektowych, które warto znać od początku

- **Zakres jest celowo wąski**: uwzględniane są wyłącznie zdarzenia transferów tokenów ERC-20 i oceniane są tylko
  konta zewnętrzne (EOA). Oba te uproszczenia są udokumentowane - to nie przeoczone luki.
- **Wartości transferów są normalizowane bez zewnętrznych danych cenowych**: stosowany jest odporny z-score (mediana/MAD
  logarytmu wartości), liczony osobno dla każdego kontraktu tokena. Jest to podejście samoodniesieniowe - wyliczane
  na nowo z dostępnych danych zamiast polegania na utrzymywanej liście cen - dzięki czemu jest odporne
  na dryf tokenów i rynku w czasie.
- **Klasyfikator to zespół specjalistów od poszczególnych rodzin cech** (wartość, gas, zachowanie,
  struktura), a nie jeden model na wszystkich cechach. Dzięki temu żadna rodzina cech nie dominuje wyniku,
  a przy okazji otrzymujemy przybliżony, pomocniczy sygnał (stopień niezgodności między rodzinami).
- **Każdy skrypt, który może korzystać z płatnych zasobów chmurowych, domyślnie działa w trybie próbnym (dry run).**
  Faktyczne pobranie danych lub wykonanie zapytania do BigQuery zawsze wymaga jawnego podania flagi "--execute".

## Znane ograniczenia

- Historyczne dane treningowe pochodzą z jednego incydentu. Ich rozkład etykiet (~26% oznaczonych jako
  powiązane z oszustwem) oraz zachowanie adresów nie odzwierciedlają rzeczywistego ruchu w sieci Ethereum, dlatego miękkie
  etykiety traktowane są jako względny ranking ryzyka, a nie skalibrowane prawdopodobieństwo.
- Notebook kontroli jakości wykazał rzeczywiste przesunięcie rozkładów między oknem historycznym a świeżymi
  danymi (np. typowy koszt gasu) - co przypomina, że wytrenowany zespół przy ocenie bieżącej aktywności
  ekstrapoluje, a nie interpoluje.
- Łączenie wyniku zespołu z ręcznie napisanymi heurystykami było częścią pierwotnego planu Etapu 1.
  Na tym etapie wykorzystywany jest wyłącznie konsensus samego zespołu, a łączenie z heurystykami
  zostało na razie porzucone jako zbędna komplikacja.

## Szybki start

```bash
uv sync
```

Skrypty pipeline'u w kolejności (każdy z tych, które korzystają z BigQuery, wyświetla szacunkowy koszt i nie robi
nic więcej, dopóki nie podasz "--execute"):

```bash
uv run python src/data/bq_fetch_day.py            # pobiera świeże dane o transferach
uv run python src/data/bq_contracts_fresh.py       # oznacza, które świeże adresy są kontraktami
uv run python src/data/sample_fresh_addresses.py   # usuwa duplikaty + losuje podzbiór adresów
uv run python src/features/build_features.py --mode train   # historyczna, oetykietowana tabela cech
uv run python src/features/build_features.py --mode score   # świeża, nieoetykietowana tabela cech
uv run python src/models/train_xgboost_ensemble.py           # trenuje zespół, loguje do MLflow
```
