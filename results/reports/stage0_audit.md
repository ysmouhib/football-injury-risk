# Stage 0 data audit

## Per-table summary

| table | rows | from_date_min | from_date_max | end_date_min | end_date_max | null_player_id | null_season_name | null_injury_reason | null_from_date | null_end_date | null_days_missed | null_competition_id | null_minutes_played | date_of_birth_min | date_of_birth_max | null_date_of_birth | null_height | null_main_position | null_foot | transfer_date_min | transfer_date_max | null_transfer_date | null_club_id | null_season_id | null_season_league_level_level_number | date_min | date_max | null_date | null_value |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| player_injuries | 168290 | 1973-09-02 | 2026-09-28 | 1973-09-06 | 2027-09-06 | 0 | 0 | 0 | 0.0001 | 0.0075 | 0 | nan | nan | NaT | NaT | nan | nan | nan | nan | NaT | NaT | nan | nan | nan | nan | NaT | NaT | nan | nan |
| player_performances | 1878719 | NaT | NaT | NaT | NaT | 0 | 0 | nan | nan | nan | nan | 0 | 0.6231 | NaT | NaT | nan | nan | nan | nan | NaT | NaT | nan | nan | nan | nan | NaT | NaT | nan | nan |
| player_profiles | 92671 | NaT | NaT | NaT | NaT | 0 | nan | nan | nan | nan | nan | nan | nan | 1952-02-13 | 2010-10-12 | 0.0109 | 0.0000 | 0.0000 | 0.2535 | NaT | NaT | nan | nan | nan | nan | NaT | NaT | nan | nan |
| transfer_history | 1101440 | NaT | NaT | NaT | NaT | 0 | nan | nan | nan | nan | nan | nan | nan | NaT | NaT | nan | nan | nan | nan | 1900-07-01 | 3026-06-30 | 0.0005 | nan | nan | nan | NaT | NaT | nan | nan |
| team_competitions_seasons | 58247 | NaT | NaT | NaT | NaT | nan | nan | nan | nan | nan | nan | 0 | nan | NaT | NaT | nan | nan | nan | nan | NaT | NaT | nan | 0 | 0 | 0.0387 | NaT | NaT | nan | nan |
| player_market_value | 901429 | NaT | NaT | NaT | NaT | 0 | nan | nan | nan | nan | nan | nan | nan | NaT | NaT | nan | nan | nan | nan | NaT | NaT | nan | nan | nan | nan | 2003-12-14 | 2025-09-11 | 0 | 0 |

## Injuries

- rows: 168,290; players: 34,561; exact-duplicate rows: 13
- snapshot end (max from_date): 2026-09-28
- days_missed stats: mean=50.2, p50=22, p90=123, p99=365, max=5582
- spells with days_missed >= 3: 166,958 (99.2%); >= 7: 145,505 (86.5%)
- days_missed == end_date - from_date + 1 on 167,009 of 167,026 rows with an end date (end_date = last day missed)
- distinct injury_reason values: 348; reasons with < 100 occurrences: 183 covering 3.25% of rows
- rows whose season label disagrees with the onset date: 3,379 (2.0%) — labels are ignored, real dates are used

### Records per season (onset date)

| season | n_records | min_from | max_from |
|---|---|---|---|
| 2005 | 397 | 2005-07-01 | 2006-06-23 |
| 2006 | 467 | 2006-07-01 | 2007-06-30 |
| 2007 | 1664 | 2007-07-01 | 2008-06-30 |
| 2008 | 2388 | 2008-07-01 | 2009-06-29 |
| 2009 | 2122 | 2009-07-01 | 2010-06-30 |
| 2010 | 3045 | 2010-07-01 | 2011-06-30 |
| 2011 | 3474 | 2011-07-01 | 2012-06-30 |
| 2012 | 4224 | 2012-07-01 | 2013-06-30 |
| 2013 | 5132 | 2013-07-01 | 2014-06-30 |
| 2014 | 5907 | 2014-07-01 | 2015-06-30 |
| 2015 | 7214 | 2015-07-01 | 2016-06-30 |
| 2016 | 8913 | 2016-07-01 | 2017-06-30 |
| 2017 | 9946 | 2017-07-01 | 2018-06-30 |
| 2018 | 11438 | 2018-07-01 | 2019-06-30 |
| 2019 | 10252 | 2019-07-01 | 2020-06-30 |
| 2020 | 15195 | 2020-07-01 | 2021-06-30 |
| 2021 | 15216 | 2021-07-01 | 2022-06-30 |
| 2022 | 14403 | 2022-07-01 | 2023-06-30 |
| 2023 | 14712 | 2023-07-01 | 2024-06-30 |
| 2024 | 15366 | 2024-07-01 | 2025-06-30 |
| 2025 | 12408 | 2025-07-01 | 2026-06-30 |
| 2026 | 2045 | 2026-07-01 | 2026-09-28 |

## Label mapping (`configs/label_mapping.yaml`)

- events after dropping exact duplicates / missing onset: 168,259
- full reason → category / body part / tissue table: `data/audit/label_mapping_table.csv`

### Category (all events; ≥ 7 days missed)

| category | rows_all | share_all | rows_ge7 | share_ge7 |
|---|---|---|---|---|
| injury | 117772 | 0.6999 | 106582 | 0.7326 |
| unknown | 30375 | 0.1805 | 26640 | 0.1831 |
| illness | 15361 | 0.0913 | 9536 | 0.0655 |
| non_injury | 4664 | 0.0277 | 2676 | 0.0184 |
| non_sport | 87 | 0.0005 | 59 | 0.0004 |

### Body part and tissue (injury events)

| body_part | rows | share |
|---|---|---|
| unspecified | 31587 | 0.2682 |
| knee | 23241 | 0.1973 |
| ankle | 11114 | 0.0944 |
| groin_hip | 10181 | 0.0864 |
| hamstring | 9603 | 0.0815 |
| lower_leg | 8174 | 0.0694 |
| upper_limb | 5462 | 0.0464 |
| foot | 4986 | 0.0423 |
| thigh | 4663 | 0.0396 |
| trunk | 4250 | 0.0361 |
| head_neck | 3397 | 0.0288 |
| lower_limb_other | 1114 | 0.0095 |

| tissue | rows | share |
|---|---|---|
| muscle | 47400 | 0.4025 |
| unspecified | 29952 | 0.2543 |
| ligament_joint | 16263 | 0.1381 |
| contusion | 8925 | 0.0758 |
| bone | 7068 | 0.0600 |
| other | 3913 | 0.0332 |
| tendon | 3303 | 0.0280 |
| concussion | 948 | 0.0080 |

### Most frequent injury reasons with unspecified tissue

| injury_reason | body_part | count |
|---|---|---|
| Knee injury | knee | 6305 |
| Ankle injury | ankle | 5409 |
| Knee problems | knee | 3522 |
| Shoulder injury | upper_limb | 2512 |
| Foot injury | foot | 1778 |
| Back problems | trunk | 1384 |
| Leg injury | lower_limb_other | 1116 |
| Hip injury | groin_hip | 927 |
| Ankle problems | ankle | 827 |
| Back injury | trunk | 751 |
| Injury to the ankle | ankle | 644 |
| Head injury | head_neck | 628 |
| Hip problems | groin_hip | 548 |
| Toe injury | foot | 397 |
| Pubalgia | groin_hip | 312 |

## Performances

- rows: 1,878,719; players: 88,375
- grain: player × season × competition × team (season aggregates, no match-level rows)
- rows duplicated on (player, season, competition): 30,728; on (player, season, competition, team): 0 — mid-season moves within a league

## Minutes column defect and exposure estimate

`minutes_played` is missing on every row with `goals == 0`:

| goals | minutes null | minutes present |
|---|---|---|
| 0 | 1170451 | 0 |
| >= 1 | 84 | 569982 |
| missing (goalkeepers) | 23 | 138179 |

For outfield players the column holds **minutes per goal** (e.g. 48 goals × 65 = 3,120 min for a 35-game LaLiga season); for goalkeepers (`goals` empty) it holds total minutes. Total minutes are recoverable only for goalkeepers and goal scorers, so raw minutes would encode goal scoring. Appearance counts are complete; minutes are estimated from them for every row.

Estimator (least squares, no intercept, fitted on seasons <= 2013): minutes = 89.81 × full games + 72.06 × starts subbed off + 20.48 × substitute appearances

| rows | n | R2 | MAE_min | bias_min |
|---|---|---|---|---|
| outfield, seasons <= 2013 (fit) | 195447 | 0.9965 | 36.0755 | -0.5061 |
| outfield, seasons > 2013 | 374535 | 0.9962 | 36.5309 | -2.4457 |
| goalkeepers, seasons > 2013 | 89438 | 0.9996 | 8.8207 | -3.3826 |
| top-5 leagues, seasons > 2013 | 16634 | 0.9953 | 45.6882 | 4.0950 |

## Player overlap (all ids are Transfermarkt player ids; no cross-source matching needed)

- players in performances: 88,375; injuries: 34,561; profiles: 92,671; transfers: 117,944
- |performances ∩ injuries ∩ profiles|: 34,524
- injured players without a profile: 0

## Season scope evidence

- snapshot end (max injury onset): **2026-09-28**
- inputs complete: season ended ≥ 45 days before the snapshot, injury onsets reach within 45 days of the season end, and some top-5 player was in ≥ 34 matchday squads (performance scrape not cut mid-season)
- labels complete: ≥ 7-day injuries per top-5 player-season ≥ 80% of the median of the 5 complete seasons before the last one (reference 0.836)

| season | max_from | top_player_seasons | top_max_squad_games | injuries_ge7_per_player_season | ended_before_snapshot | injuries_reach_june | top_squads_complete | inputs_complete | labels_complete |
|---|---|---|---|---|---|---|---|---|---|
| 2005 | 2006-06-23 | 1960 | 38 | 0.0403 | True | True | True | True | False |
| 2006 | 2007-06-30 | 2026 | 38 | 0.0365 | True | True | True | True | False |
| 2007 | 2008-06-30 | 2150 | 38 | 0.2874 | True | True | True | True | False |
| 2008 | 2009-06-29 | 2278 | 38 | 0.4434 | True | True | True | True | False |
| 2009 | 2010-06-30 | 2381 | 38 | 0.3667 | True | True | True | True | False |
| 2010 | 2011-06-30 | 2419 | 38 | 0.5101 | True | True | True | True | False |
| 2011 | 2012-06-30 | 2485 | 38 | 0.4909 | True | True | True | True | False |
| 2012 | 2013-06-30 | 2537 | 38 | 0.6082 | True | True | True | True | False |
| 2013 | 2014-06-30 | 2603 | 38 | 0.6435 | True | True | True | True | False |
| 2014 | 2015-06-30 | 2655 | 38 | 0.7164 | True | True | True | True | True |
| 2015 | 2016-06-30 | 2756 | 38 | 0.7489 | True | True | True | True | True |
| 2016 | 2017-06-30 | 2803 | 38 | 0.8769 | True | True | True | True | True |
| 2017 | 2018-06-30 | 2803 | 38 | 0.8758 | True | True | True | True | True |
| 2018 | 2019-06-30 | 2802 | 38 | 0.9625 | True | True | True | True | True |
| 2019 | 2020-06-30 | 2976 | 38 | 0.8364 | True | True | True | True | True |
| 2020 | 2021-06-30 | 3200 | 38 | 0.9981 | True | True | True | True | True |
| 2021 | 2022-06-30 | 3332 | 38 | 0.9379 | True | True | True | True | True |
| 2022 | 2023-06-30 | 3235 | 38 | 0.7598 | True | True | True | True | True |
| 2023 | 2024-06-30 | 3192 | 38 | 0.8064 | True | True | True | True | True |
| 2024 | 2025-06-30 | 3160 | 38 | 0.8608 | True | True | True | True | True |
| 2025 | 2026-06-30 | 2183 | 4 | 0.8722 | True | True | False | False | True |
| 2026 | 2026-09-28 | nan | nan | nan | False | False | False | False | False |

**Derived scope:** seasons 2014/15 .. 2024/25; last complete season 2024/25 is the final holdout.

## Panel membership: registration vs appearance records

Registration (last transfer dated ≤ d) is knowable at d; appearance records depend on the whole season. Records not covered by senior-team registration are mostly youth/reserve registrations (B teams, U19/U23) with few first-team games; registered players without a record were never in a matchday squad (e.g. injured all season) and are kept.

| season | with_top5_appearance_record | registered_oct15_or_mar15 | records_covered_by_registration | registered_without_record |
|---|---|---|---|---|
| 2014 | 2655 | 2610 | 0.8757 | 0.1092 |
| 2015 | 2756 | 2680 | 0.8846 | 0.0903 |
| 2016 | 2803 | 2670 | 0.8776 | 0.0787 |
| 2017 | 2803 | 2670 | 0.8826 | 0.0734 |
| 2018 | 2802 | 2691 | 0.8808 | 0.0829 |
| 2019 | 2976 | 2711 | 0.8552 | 0.0612 |
| 2020 | 3200 | 2793 | 0.8137 | 0.0677 |
| 2021 | 3332 | 2790 | 0.7971 | 0.0480 |
| 2022 | 3235 | 2791 | 0.8247 | 0.0441 |
| 2023 | 3192 | 2698 | 0.8086 | 0.0434 |
| 2024 | 3160 | 2735 | 0.8237 | 0.0483 |
