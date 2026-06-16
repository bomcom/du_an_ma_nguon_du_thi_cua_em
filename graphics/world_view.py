# graphics/world_view.py

import pygame
import logging

logger = logging.getLogger("WorldViewEngine")


class WorldView:

    def __init__(self,
                 width=1200,        # Đảm bảo có dấu phẩy ở cuối dòng này
                 height=900,        # Đảm bảo có dấu phẩy ở cuối dòng này
                 grid_size=None,    # Đảm bảo có dấu phẩy ở cuối dòng này
                 **kwargs):
        
        pygame.init()
        
        self.width = width
        self.height = height
        
        self.screen = pygame.display.set_mode(
            (width, height)
        )
        
        pygame.display.set_caption(
            "LHU Interdisciplinary World Engine"
        )
        
        self.clock = pygame.time.Clock()
        
        self.font = pygame.font.SysFont(
            "Consolas",
            16
        )
        
        self.font_big = pygame.font.SysFont(
            "Consolas",
            22,
            bold=True
        )
        
        logger.info(
            "[WorldView] Initialized"
        )

    def process_user_events(self):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                return False

        return True

    def draw_world(self, registry):
        """Render world with typed entities: Prey (light blue), Predators (red), Flora (green)."""
        world_rect = pygame.Rect(
            0,
            0,
            900,
            self.height
        )

        pygame.draw.rect(
            self.screen,
            (30, 50, 30),
            world_rect
        )

        entities = registry.query("Transform")

        for eid in entities:

            transform = registry.get_component_snapshot(
                eid,
                "Transform"
            )

            if not transform:
                continue

            x = int(transform.get("x", 0))
            y = int(transform.get("y", 0))

            x %= 880
            y %= 880

            # Determine entity type by mass and components
            mass_comp = registry.get_component_snapshot(eid, "Mass")
            mass = mass_comp.get("mass_kg", 2.0) if mass_comp else 2.0
            
            flora_comp = registry.get_component_snapshot(eid, "FloraDef")
            brain_comp = registry.get_component_snapshot(eid, "NeuralBrain")
            
            # Classify: Flora if no mass/brain, Predator if heavy brain, else Prey
            if flora_comp or (mass_comp and mass < 1.0):
                # Flora: green, small
                color = (0, 180, 60)
                radius = 3
            elif brain_comp:
                brain_hash = brain_comp.get("weight_hash", 0.0)
                if brain_hash > 0.3:  # Predator signature
                    color = (255, 80, 80)  # Red
                    radius = 7
                else:
                    color = (100, 200, 255)  # Light blue
                    radius = 5
            else:
                color = (100, 200, 255)  # Default prey color
                radius = 5

            pygame.draw.circle(
                self.screen,
                color,
                (x + 10, y + 10),
                radius
            )

    def draw_hud(
        self,
        registry,
        math_state,
        gatekeeper,
        ltspice
    ):
        """Enhanced HUD with full system state including energy, mass, dynamics, and violations."""
        hud_x = 900

        pygame.draw.rect(
            self.screen,
            (25, 25, 25),
            (hud_x, 0, 300, self.height)
        )

        y = 20

        title = self.font_big.render(
            "SYSTEM STATUS",
            True,
            (255, 255, 255)
        )

        self.screen.blit(
            title,
            (hud_x + 20, y)
        )

        y += 50

        # Extract metrics
        entities = registry.entity_count()
        prey_count = int(math_state.get('prey', 0))
        pred_count = int(math_state.get('predator', 0))
        e_total = math_state.get('E_total', 0.0)
        e_max = math_state.get('E_max', 1000.0)
        m_total = math_state.get('M_total', 0.0)
        m_max = math_state.get('M_max', 5000.0)
        v_max = math_state.get('v_max', 0.0)
        lorenz = math_state.get('lorenz', [0, 0, 0])
        decay = math_state.get('decay_trigger', 0)
        viols = list(math_state.get('violations', {}).keys())
        bw = ltspice.get('bandwidth', 0.0)
        snr = ltspice.get('snr', 0.0)

        telemetry = [
            f"Entities : {entities}",
            f"Prey     : {prey_count}",
            f"Predator : {pred_count}",
            "",
            f"E: {e_total:.0f}/{e_max:.0f}",
            f"M: {m_total:.0f}/{m_max:.0f}",
            f"v_max    : {v_max:.2f}",
            "",
            f"Lorenz X : {lorenz[0]:.2f}",
            f"Lorenz Y : {lorenz[1]:.2f}",
            f"Lorenz Z : {lorenz[2]:.2f}",
            "",
            f"Decay    : {decay}",
            f"Violations: {len(viols)}",
            f"  {', '.join(viols[:2]) if viols else 'None'}",
            "",
            f"BW       : {bw:.0f}",
            f"SNR      : {snr:.1f}",
            "",
            f"Gate     : {gatekeeper}",
            "",
            f"FPS      : {self.clock.get_fps():.1f}",
        ]

        for line in telemetry:

            surf = self.font.render(
                line,
                True,
                (220, 220, 220)
            )

            self.screen.blit(
                surf,
                (hud_x + 20, y)
            )

            y += 24

    def render_frame(
        self,
        registry,
        math_state,
        gatekeeper,
        ltspice
    ):
        """Render single frame with world and HUD."""
        self.screen.fill(
            (10, 10, 10)
        )

        self.draw_world(registry)

        self.draw_hud(
            registry,
            math_state,
            gatekeeper,
            ltspice
        )

        pygame.display.flip()

        self.clock.tick(60)

    def run(self, app):

        logger.info(
            "[WorldView] Rendering Loop Started"
        )

        running = True

        while running:

            running = self.process_user_events()

            snapshot, version = (
                app.box.read_render_snapshot()
            )

            math_state = snapshot.get(
                "math_state",
                {}
            )

            ltspice = snapshot.get(
                "ltspice",
                {}
            )

            flags = snapshot.get(
                "adversarial_flags",
                {}
            )

            gatekeeper = (
                "VIOLATION"
                if flags.get("is_violation", False)
                else "PASS"
            )

            self.render_frame(
                app.registry,
                math_state,
                gatekeeper,
                ltspice
            )

        pygame.quit()

