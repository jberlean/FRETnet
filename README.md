# FRETnet

## Quickstart
Currently `sim_network.py` and `sim_network_nogui.py` simulate a 2-input AND gate. They can be run with the syntax:
```
python -i sim_network.py <X1> <X2>
```
or
```
python -i sim_network_nogui.py <X1> <X2>
```
where `<X1>` and `<X2>` are either `ON` or `OFF`.

`sim_network.py` prints out the activation of the output node every minute. `sim_network_nogui.py` prints out the activation of the output node every 10000 steps.

## GUI controls

* `<Enter>`: advance the simulation 1 step
* `<Space>`: start/stop continuous simulation mode
* `<Left>`: slow down continuous simulation
* `<Right>`: speed up continuous simulation

While in GUI mode, each node is displayed with its activation (time-averaged flux) shown.

## FRETnet components

### `utils.py`

`Node` objects define individual fluorophores.
Each `Node` has a `decay_rate` and `emit_rate` describing its intrinsic non-readiative decay and radiative emission rate constants, respectively
(note that activation values do not distinguish between these two decay pathways). 
The `InputNode` object inherits from `Node`, and has an additional `production_rate` parameter describing its excitation rate constant via external power source.
Two `Nodes` `n1` and `n2` may be linked in an input-output relationship via `n2.add_input(n1)`

A `Network` object simulations a collection of nodes. Note that for GUI simulations, the `Network` object need not be explicitly generated (it is created by the GUI).
Each `Network` object collects statistics on its nodes' behavior, as well as the global simulation time.

For `Network` `network`, the simulation time is in `network._time`, and node statistics are stored in a `dict` (`network._stats`). `network._stats` has `tuple` keys following the form `(rxn_type, input, output)`:
* `rxn_type`: the type of reaction (`emit`, `decay`, `production`, `transfer`, or the wildcard `'*'`)
* `input`: the input node (a `Node` object, `None`, or the wildcard `'*'`)
* `output`: the output node (a `Node` object, `None`, or the wildcard `'*'`)
The value of each `dict` entry is a positive integer representing the number of times the reaction occurred over the course of the simulation.

### `gui.py`

The `Simulator` object generates a Tkinter window that simulates a given collection of nodes with display positions specified. See `sim_network.py` for example usage.

