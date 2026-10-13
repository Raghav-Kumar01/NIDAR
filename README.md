# NIDAR team repository

## AirMouse opening and human perception

The combined perception implementation is isolated in **[AirMouse_Opening_AI](AirMouse_Opening_AI/)**.

- **[Full combined application: setup and commands](AirMouse_Opening_AI/docs/FULL_APPLICATION.md)**
- **[Team setup and Windows CMD instructions](AirMouse_Opening_AI/docs/TEAM_SETUP.md)**
- [Architecture, result contract and measured integration evidence](AirMouse_Opening_AI/docs/PERCEPTION_INTEGRATION.md)
- [Opening-model limitations](AirMouse_Opening_AI/docs/MODEL_CARD.md)

This is an experimental research integration. Both systems process the same frame and return opening polygons and person boxes. The opening baseline failed external-image semantic checks; it is not ready for live flight decisions. Full mode adds tracking and persistent appearance-based identity estimates; these are not guaranteed real-world identities.

Source code lives in Git. The opening checkpoint is a versioned release asset; setup fetches the pinned upstream human repository and verifies checkpoint hashes. Training datasets are not included.
