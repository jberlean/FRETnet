import examples.networks as nets
import analysis.analyze as analyze

lin2 = nets.linear()
probs2 = analyze.probability_by_network_state(lin2)

lin3 = nets.linear(n=3)
probs3 = analyze.probability_by_network_state(lin3)

lin4 = nets.linear(n=4)
probs4 = analyze.probability_by_network_state(lin4)

lin7 = nets.linear(n=7)
probs7 = analyze.probability_by_network_state(lin7)
fluxes7 = analyze.node_fluxes(lin7)
