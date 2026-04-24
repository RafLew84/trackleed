# Tracker Integration Guide

Ten przewodnik opisuje minimalny zestaw zmian potrzebnych do dodania kolejnego backendu trackingowego po refaktoryzacji warstwy trackingowej.

Nie zaklada on, ze rejestr jest juz jedynym zrodlem prawdy. Dzis koszt integracji jest duzo nizszy niz przed refaktorem, ale nadal sa 2 typy zmian:
- rdzen backendu i rejestr
- lekkie mapy GUI / visualizera

## Architektura po refaktorze

Najwazniejsze miejsca:
- `trackleed/analysis/`
  - `*_backend.py`: probe srodowiska, raport, runtime settings, helpery modelu
  - `tracking_point_backends.py`: publiczne `track_spots_*` dla point trackerow
  - `tracking_bbox_backends.py`: publiczne `track_spots_*` dla bbox trackerow
  - `tracking_mask_backends.py`: publiczne `track_spots_*` dla mask trackerow
  - `tracking_retrack.py`: publiczne `retrack_single_spot_*`
- `trackleed/gui/tracker_registry.py`
  - `TrackerSpec`
  - glowny rejestr backendow widocznych w GUI
- `trackleed/gui/main_window.py`
  - panel i settings
  - `_collect_*_tracking_params()`
  - refresh checkpointow / diagnostyki
- `trackleed/gui/tracking_dialog.py`
  - etykiety metod i szczegoly backendu
  - retrack z visualizera
- `trackleed/tests/`
  - testy kontraktowe rejestru, dispatchu i shimow

## Minimalny przeplyw dodania nowego trackera

### 1. Dodaj runtime backend

W `trackleed/analysis/` dodaj modul, np.:
- `deva_backend.py`
- `ostrack_backend.py`

Minimalny kontrakt:
- `probe_<name>_environment(...)`
- `format_<name>_environment_report(status)`
- dataclass settings / runtime settings, jesli backend ich potrzebuje
- helpery wyboru checkpointu / wariantu, jesli backend ma wiele modeli

Jesli backend ma subprocess:
- dodaj worker `scripts/run_<name>_subprocess.py`
- dodaj checker `scripts/check_<name>_env.py`

### 2. Wystaw publiczne entrypointy trackingu

Umiesc glowny entrypoint w odpowiednim module rodzinnym:
- point tracker -> `tracking_point_backends.py`
- bbox tracker -> `tracking_bbox_backends.py`
- mask tracker -> `tracking_mask_backends.py`

Nazewnictwo:
- `track_spots_<name>_offline(...)`
- `track_spots_<name>_online(...)` jesli potrzeba

Jesli tracker wspiera retrack:
- dodaj `retrack_single_spot_<name>_offline(...)` do `tracking_retrack.py`

## 3. Dodaj wpis do `TrackerSpec`

W `trackleed/gui/tracker_registry.py` dopisz nowy `TrackerSpec`.

Pola krytyczne:
- `tracker_id`
- `display_name`
- `scenario_name`
- `family`
- `source_names`
- `probe_environment`
- `format_environment_report`
- `collect_params_method_name`
- `run_tracking`
- `run_retrack`
- `supports_direct`
- `supports_subprocess`

Pola GUI, ktore zwykle tez trzeba wypelnic:
- `settings_attr_name`
- `params_argument_name`
- `panel_group_attr`
- `warning_label_attr`
- `backend_mode_widget_attr` dla subprocess
- `repo_path_attr` dla repo zewnetrznego
- `python_executable_attr` dla dedykowanego env
- `checkpoint_selector_attr` jesli checkpoint jest wybierany w GUI
- `variant_selector_attr` jesli backend ma wiele wariantow

Wazne:
- `source_names` musza pokrywac realne wartosci `spot.source`
- jesli jeden runtime obsluguje kilka wariantow zrodlowych, zarejestruj je wszystkie, jak `tomp_offline` / `tamos_offline`

## 4. Dodaj settings i collect params w `MainWindow`

Minimum po stronie GUI:
- settings dataclass lub reuse istniejacej
- grupa widgetow, jesli backend ma osobny panel
- `_collect_<name>_tracking_params()`

Jesli backend ma panel:
- utworz `self.<name>_group`
- utworz `self.<name>_warning_label`
- dopnij pola zgodne z `TrackerSpec`

Jesli backend ma checkpointy lub warianty:
- dodaj refresh listy checkpointow
- dopnij logike wyboru domyslnego checkpointu

Po refaktorze `MainWindow` korzysta juz z rejestrowych helperow dla:
- `probe/report`
- synchronizacji `settings <- widgets`
- walidacji `direct/subprocess`
- `run_tracking`

To znaczy, ze nowy backend nie powinien dokladac kolejnego duzego `if/elif`, tylko glownie:
- collect params
- panel / refresh

## 5. Dopnij visualizer i retrack

W `trackleed/gui/tracking_dialog.py` zwykle trzeba dopisac:
- `TRACKER_METHOD_LABEL_BY_SOURCE`
- `MANUAL_AUGMENTED_METHOD_LABEL_BY_SOURCE`

Jesli tracker wspiera retrack z visualizera:
- `TRACKER_VISUALIZER_RETRACK_METHOD_NAME_BY_ID`
- `TRACKER_UNAVAILABLE_TITLE_BY_ID`
- `TRACKER_ENV_HINT_BY_ID` dla backendu subprocessowego
- `initiate_retrack_<name>(...)`

Jesli chcesz pokazywac szczegoly backendu:
- dopisz sekcje do `POINT_TRACKER_DETAIL_SPECS`
- albo odpowiednie detail helpery dla `bbox` / `mask`

Jesli tracker ma wlasne raw pointy / bbox / maske:
- dopnij overlay focus pointow albo pola diagnostyczne

## 6. Kiedy nowy panel GUI nie jest potrzebny

Nie kazdy tracker musi miec od razu pelny panel.

Mozesz zaczac od backendu bez nowego panelu, jesli:
- reuse’ujesz istniejace settings
- checkpoint i wariant sa stale
- backend nie ma wielu trybow runtime

Wtedy w `TrackerSpec` mozna ograniczyc liczbe pol GUI, ale:
- `collect_params_method_name`
- `params_argument_name`
- `panel_group_attr`
- `warning_label_attr`

nadal powinny pozostac spojne z kontraktem testow.

## 7. Testy, ktore trzeba uruchomic

Minimum po dodaniu nowego backendu:

```bash
python3 -m unittest trackleed.tests.test_tracker_registry
python3 -m unittest trackleed.tests.test_tracker_refactor_readiness
python3 -m unittest trackleed.tests.test_main_window_registry_dispatch
python3 -m unittest trackleed.tests.test_tracking_dialog_retrack_dispatch
python3 -m unittest trackleed.tests.test_tracking_shim_compatibility
```

Dodatkowo:
- checker env, np. `scripts/check_<name>_env.py`
- manualny smoke test w GUI
- jesli backend ma subprocess, test w dedykowanym env

## 8. Checklista gotowosci PR

Przed uznaniem integracji za domknieta sprawdz:
- backend ma `probe + format report`
- `TrackerSpec` istnieje i ma poprawne `source_names`
- `MainWindow` ma `_collect_*_tracking_params()`
- visualizer zna etykiete metody
- retrack dziala albo jest jawnie wylaczony
- testy kontraktowe przechodza
- manualny smoke test przechodzi

## 9. Co jeszcze nie jest w pelni deklaratywne

Po refaktorze najwieksze pozostale jawne mapy sa w:
- `trackleed/gui/tracking_dialog.py`
  - etykiety metod
  - mapowanie `tracker_id -> retrack handler`
- `trackleed/gui/main_window.py`
  - czesc refreshu checkpointow / wariantow
- komunikaty `... Unavailable` i hinty env

To nie blokuje nowego backendu, ale oznacza, ze przy integracji nadal trzeba sprawdzic kilka map poza samym `TrackerSpec`.

## 10. Rekomendowana kolejnosc dla nowego trackera

Najbezpieczniej:
1. backend runtime + checker env
2. `track_spots_*` + `retrack_single_spot_*`
3. `TrackerSpec`
4. collect params + panel GUI
5. visualizer labels / retrack
6. testy kontraktowe
7. smoke test manualny

Ta kolejnosc ogranicza ryzyko, ze UI bedzie gotowe szybciej niz stabilny runtime.
