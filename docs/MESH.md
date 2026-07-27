# Self-healing sensor mesh (TLK-MESH)

A Terra site is a low-power wireless mesh of TLK nodes reporting to one or more
gateways. It heals two ways: packets reroute when a link drops, and data is
imputed when a node drops. The second is the part competitors don't have,
because it comes from the engine, not the radio.

See `terra/mesh.py` for the model this design is tested against, and FIG. 07 for
the diagram.

## Why a mesh

Real farms, wetlands, and habitats are RF-hostile and often have no reliable
uplink. A star topology dies when its one link dies. A mesh gives every node
multiple paths to a gateway, extends range through multi-hop, and keeps working
when part of the site goes dark. Autonomy cannot depend on the cloud, so the
network has to stand on its own.

## Layers

**RF.** Sub-GHz (868/915 MHz, LoRa-class) for range and penetration, or 2.4 GHz
802.15.4 / Thread for density and throughput. Chosen per site; the upper layers
don't care which.

**Routing.** Each node keeps a least-cost route to the nearest gateway, cost
being the sum of `-log(link quality)` so chaining good links stays cheap, and a
backup next-hop. This is a distance-vector / DODAG structure — no central
controller, so there is no single point of failure.

**Heal.** Nodes beacon periodically. A missed-beacon count trips local route
repair: switch to the backup next-hop, or rediscover. In the model,
`routes()` is simply recomputed after a failure; on hardware this is event-driven
and local. Multiple gateways give uplink redundancy.

**Sync.** A gossip/flood time-sync keeps sampling coordinated across the mesh so
readings are comparable and the engine's spatial model is valid.

**Buffer.** When a node is partitioned from every gateway it buffers and
store-and-forwards, replaying on reconnection. This already exists in the node
runtime; the mesh just decides when a path is available.

**Security.** Per-node keys and authenticated, encrypted frames, provisioned
through the same enrollment-token flow as the platform. A node joins the mesh
the same way it enrolls with the fleet.

**Power.** Duty-cycled radios on solar/battery. The engine is numpy-only with no
GPU, so a node senses, decides, and acts inside a tight power budget.

## Sensing self-heal — the differentiator

Standard mesh heals packets. Terra also heals data. Because the engine models a
spatially coupled field (dissolved oxygen, temperature, nutrient load are smooth
across a site), a dropped node's reading is not simply missing — it is inferred
from its still-connected neighbours, quality- and distance-weighted, until the
node returns. `MeshNetwork.impute()` does this; in the test, a dropped node is
reconstructed to within ~0.06 of truth and the whole-fleet mean error stays
under 0.25 on the modelled field.

The same idea runs deeper in the engine itself: partial observability lets the
UKF keep estimating hidden state from the surviving channels, and the fleet
shared prior supplies what a brand-new or long-silent node hasn't learned yet.
So a fouled probe (bias tracking), a dead channel (partial observability), and a
dead node (neighbour imputation) are three faces of one property — the site
keeps a calibrated estimate everywhere, even where it is currently blind.

## Status

The routing and imputation behaviour is modelled and tested in `terra/mesh.py`
and `tests/test_mesh.py`. The RF layer, time-sync, and hardware integration are
a design for the node partnership, not yet built. These numbers firm up on real
radios and a real site.
