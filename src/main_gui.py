import pygame
import pygame_gui
import sys
import numpy as np
import glob
import os

from helper import getSafeWhiteCoords
from simulation import Simulation
import loader as ld

# --- Configuration Constants ---
SIDEBAR_WIDTH = 320
UI_FPS = 60
MIN_WIN_WIDTH = 1200
MIN_WIN_HEIGHT = 850

# Colors
COLOR_BG = (30, 30, 30)
COLOR_MAP_BG = (15, 15, 15)
COLOR_WALL = (200, 200, 200)
COLOR_EXIT = (0, 255, 0)
COLOR_AGENT = (255, 50, 50)
WALL_CHARS = {'#', '1', 'W', '@'}

class SimulationGUI:
    def __init__(self):
        pygame.init()
        
        # --- File Discovery ---
        self.available_floorplans = []
        fplan_files = glob.glob("data/floorPlans/*.fplan")
        for f in fplan_files:
            basename = os.path.splitext(os.path.basename(f))[0]
            self.available_floorplans.append(basename)
        self.available_floorplans.sort()
        
        # Default to freihausEG or first available
        default_floor = "freihausEG"
        self.current_floor_name = default_floor if default_floor in self.available_floorplans else self.available_floorplans[0]

        # --- Window Setup ---
        self.window_size = (MIN_WIN_WIDTH, MIN_WIN_HEIGHT)
        self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
        pygame.display.set_caption("Escape Simulation Configurator")
        
        self.ui_manager = pygame_gui.UIManager(self.window_size)
        self.clock = pygame.time.Clock()

        # --- Simulation State ---
        self.sim = None
        self.paused = True
        self.step_once = False
        self.sim_accumulated_time = 0.0
        self.target_sim_fps = 30.0
        self.current_seed = 42
        
        # Layout State
        self.cell_size = 10
        self.map_offset_x = 0
        self.map_offset_y = 0

        # --- Load & Start ---
        self.load_floor_data(self.current_floor_name)
        self.setup_ui()
        self.reinit_simulation()

    def load_floor_data(self, floor_name):
        print(f"Loading data for: {floor_name}")
        floor_path = f"data/floorPlans/{floor_name}.fplan"
        sff_main_path = f"data/floorPlansSSF/{floor_name}_sff.npy"
        sff_pattern = f"data/floorPlansSSF/{floor_name}_sff_*.npy"

        try:
            self.floor_layout = ld.loadFloorPlan(floor_path)
            self.sff_all = ld.load_sff_from_npy(sff_main_path)
            self.sff_list = [ld.load_sff_from_npy(x) for x in glob.glob(sff_pattern)]
            self.current_floor_name = floor_name
        except Exception as e:
            print(f"Error loading {floor_name}: {e}")
            return

        self.calculate_layout_metrics()
        self.render_static_map()

    def calculate_layout_metrics(self):
        """
        Determines cell size and offsets to center/fit the map 
        in the area to the left of the sidebar.
        """
        map_rows, map_cols = self.floor_layout.shape
        
        # Available space for map
        avail_w = self.window_size[0] - SIDEBAR_WIDTH - 20 # 20px padding
        avail_h = self.window_size[1] - 20
        
        # Calculate max cell size that fits dimensions
        scale_w = avail_w / map_cols
        scale_h = avail_h / map_rows
        
        # Pick the smaller scale to fit whole map, limit min/max size
        self.cell_size = min(scale_w, scale_h)
        self.cell_size = max(2, int(self.cell_size)) # at least 2px
        
        # Calculate centering offsets
        pixel_map_w = map_cols * self.cell_size
        pixel_map_h = map_rows * self.cell_size
        
        self.map_offset_x = (avail_w - pixel_map_w) // 2 + 10
        self.map_offset_y = (avail_h - pixel_map_h) // 2 + 10

        # Resize background surface
        self.background_surface = pygame.Surface((pixel_map_w, pixel_map_h))

    def render_static_map(self):
        self.background_surface.fill(COLOR_MAP_BG)
        rows, cols = self.floor_layout.shape
        
        for r in range(rows):
            for c in range(cols):
                rect = (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size) # Note: x=col, y=row
                
                # Check walls
                if self.floor_layout[r, c] in WALL_CHARS:
                    pygame.draw.rect(self.background_surface, COLOR_WALL, rect)
                
                # Check exits (SFF == 0)
                elif self.sff_all[r, c] == 0:
                     pygame.draw.rect(self.background_surface, COLOR_EXIT, rect)

    def setup_ui(self):
        self.ui_manager.clear_and_reset()
        
        # Sidebar positioning
        x_start = self.window_size[0] - SIDEBAR_WIDTH + 10
        y = 10
        w = SIDEBAR_WIDTH - 20

        def space(pixels=35): nonlocal y; y += pixels

        # --- Controls ---
        pygame_gui.elements.UILabel(pygame.Rect((x_start, y), (w, 30)), "CONTROLS", self.ui_manager)
        space(40)

        # Floorplan
        pygame_gui.elements.UILabel(pygame.Rect((x_start, y), (w, 20)), "Select Floorplan:", self.ui_manager)
        space(20)
        self.drop_floor = pygame_gui.elements.UIDropDownMenu(
            options_list=self.available_floorplans,
            starting_option=self.current_floor_name,
            relative_rect=pygame.Rect((x_start, y), (w, 30)),
            manager=self.ui_manager
        )
        space(40)

        pygame_gui.elements.UILabel(pygame.Rect((x_start, y), (w, 20)), "Simulation Seed (Integer):", self.ui_manager)
        space(20)
        self.entry_seed = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((x_start, y), (w, 30)),
            manager=self.ui_manager
        )
        self.entry_seed.set_text(str(self.current_seed))
        self.entry_seed.set_allowed_characters('numbers') 
        space(40)

        # Agent Count
        pygame_gui.elements.UILabel(pygame.Rect((x_start, y), (w, 20)), "Agent Count:", self.ui_manager)
        space(20)
        self.slider_agents = pygame_gui.elements.UIHorizontalSlider(
            pygame.Rect((x_start, y), (w, 25)), start_value=100, value_range=(1, 700), manager=self.ui_manager
        )
        self.lbl_agents = pygame_gui.elements.UILabel(pygame.Rect((x_start+w-50, y-20), (50, 20)), "100", self.ui_manager)
        space(40)

        # Params
        pygame_gui.elements.UILabel(pygame.Rect((x_start, y), (w, 20)), "Greediness (k):", self.ui_manager)
        space(20)
        self.slider_k = pygame_gui.elements.UIHorizontalSlider(
            pygame.Rect((x_start, y), (w, 25)), start_value=5, value_range=(0, 20), manager=self.ui_manager
        )
        self.lbl_k = pygame_gui.elements.UILabel(pygame.Rect((x_start+w-50, y-20), (50, 20)), "5.0", self.ui_manager)
        space(40)

        pygame_gui.elements.UILabel(pygame.Rect((x_start, y), (w, 20)), "Xi (Wall Repulsion):", self.ui_manager)
        space(20)
        self.slider_xi = pygame_gui.elements.UIHorizontalSlider(
            pygame.Rect((x_start, y), (w, 25)), start_value=0.5, value_range=(0.0, 1.0), manager=self.ui_manager
        )
        self.lbl_xi = pygame_gui.elements.UILabel(pygame.Rect((x_start+w-50, y-20), (50, 20)), "0.5", self.ui_manager)
        space(40)

        pygame_gui.elements.UILabel(pygame.Rect((x_start, y), (w, 20)), "Speed (Steps/Sec):", self.ui_manager)
        space(20)
        self.slider_speed = pygame_gui.elements.UIHorizontalSlider(
            pygame.Rect((x_start, y), (w, 25)), start_value=30, value_range=(1, 100), manager=self.ui_manager
        )
        self.lbl_speed = pygame_gui.elements.UILabel(pygame.Rect((x_start+w-50, y-20), (50, 20)), "30", self.ui_manager)
        space(40)

        # Flags 
        # Order must match the arguments in Simulation.__init__
        self.feature_names = ["Personalized Exit", "Cluster Spawn", "Reduced Agent Mobility", "Greedy K"]
        self.flag_buttons = {}
        
        pygame_gui.elements.UILabel(pygame.Rect((x_start, y), (w, 20)), "Enable Features:", self.ui_manager)
        space(25)
        
        for name in self.feature_names:
            # Start UNCHECKED (False)
            btn = pygame_gui.elements.UIButton(pygame.Rect((x_start, y), (w, 30)), f"[ ] {name}", self.ui_manager)
            btn.user_data = {"name": name, "checked": False}
            self.flag_buttons[btn] = name
            space(35)
        
        space(10)
        self.btn_restart = pygame_gui.elements.UIButton(pygame.Rect((x_start, y), (w, 40)), "APPLY & RESTART", self.ui_manager)
        space(50)
        
        # Playback
        self.btn_step = pygame_gui.elements.UIButton(pygame.Rect((x_start, y), (w/2 - 5, 40)), "STEP", self.ui_manager)
        self.btn_play = pygame_gui.elements.UIButton(pygame.Rect((x_start + w/2 + 5, y), (w/2 - 5, 40)), "PLAY/PAUSE", self.ui_manager)
        space(50)

        # Stats Panel
        pygame_gui.elements.UILabel(pygame.Rect((x_start, y), (w, 25)), "METRICS", self.ui_manager)
        space(25)
        self.lbl_status = pygame_gui.elements.UILabel(pygame.Rect((x_start, y), (w, 25)), "Status: Ready", self.ui_manager)
        space(25)
        self.lbl_active = pygame_gui.elements.UILabel(pygame.Rect((x_start, y), (w, 25)), "Active: 0", self.ui_manager)
        space(25)
        self.lbl_exited = pygame_gui.elements.UILabel(pygame.Rect((x_start, y), (w, 25)), "Exited: 0", self.ui_manager)
        space(25)
        self.lbl_steps = pygame_gui.elements.UILabel(pygame.Rect((x_start, y), (w, 25)), "Steps: 0", self.ui_manager)
        space(25)
        self.lbl_collisions = pygame_gui.elements.UILabel(pygame.Rect((x_start, y), (w, 25)), "Collisions: 0", self.ui_manager)
        space(25)
        self.lbl_blocked = pygame_gui.elements.UILabel(pygame.Rect((x_start, y), (w, 25)), "Blocked: 0", self.ui_manager)

    def reinit_simulation(self):
        # 1. Parse Seed
        try:
            raw_text = self.entry_seed.get_text()
            seed_val = int(raw_text)
            self.current_seed = seed_val
        except ValueError:
            print(f"Invalid seed '{raw_text}', reverting to {self.current_seed}")
            self.entry_seed.set_text(str(self.current_seed))
            seed_val = self.current_seed

        print(f"Initializing RNG with seed: {seed_val}")
        self.rng = np.random.default_rng(seed_val)

        # 2. Calculate Capacity
        max_cap = len(getSafeWhiteCoords(self.floor_layout, self.sff_all))
        desired = int(self.slider_agents.get_current_value())
        
        # Clamp value
        if desired > max_cap:
            print(f"Clamping agents from {desired} to {max_cap} (Map Capacity)")
            desired = max_cap
            self.slider_agents.set_current_value(desired)
            self.lbl_agents.set_text(str(desired))

        # 3. Parameters
        k_val = self.slider_k.get_current_value()
        xi_val = self.slider_xi.get_current_value()
        
        # 4. Flags Logic
        # The UI represents "Enabled" (True).
        # The Simulation expects "Disabled" (True).
        disable_flags = []
        for btn, name in self.flag_buttons.items():
            is_enabled = btn.user_data['checked']
            disable_flags.append(not is_enabled)

        # 5. Init
        try:
            self.sim = Simulation(
                self.rng,
                self.floor_layout,
                self.sff_all,
                self.sff_list,
                desired,
                k_val,
                xi_val,
                *disable_flags
            )
            self.paused = True
            self.lbl_status.set_text("Status: Ready (Paused)")
            self.update_stats()
        except Exception as e:
            print(f"Init Error: {e}")

    def update_stats(self):
        if self.sim:
            active = len(self.sim.agentmap.agents)
            finished = len(self.sim.agentmap.finished_agents)
            
            metrics = self.sim.metrics
            
            self.lbl_active.set_text(f"Active: {active}")
            self.lbl_exited.set_text(f"Exited: {finished}")
            self.lbl_steps.set_text(f"Steps: {metrics.steps_taken}")
            self.lbl_collisions.set_text(f"Collisions: {metrics.collisions}")
            self.lbl_blocked.set_text(f"Blocked: {metrics.blocked}")

    def run(self):
        while True:
            time_delta = self.clock.tick(UI_FPS) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                
                # Handle window resizing
                if event.type == pygame.VIDEORESIZE:
                    self.window_size = event.size
                    self.ui_manager.set_window_resolution(self.window_size)
                    self.calculate_layout_metrics()
                    self.render_static_map()
                    self.setup_ui()

                self.ui_manager.process_events(event)

                if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                    if event.ui_element == self.drop_floor:
                        self.load_floor_data(event.text)
                        self.reinit_simulation()

                elif event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.btn_play:
                        self.paused = not self.paused
                        self.lbl_status.set_text("Running..." if not self.paused else "Paused")
                    elif event.ui_element == self.btn_step:
                        self.step_once = True; self.paused = True
                        self.lbl_status.set_text("Stepped")
                    elif event.ui_element == self.btn_restart:
                        self.reinit_simulation()
                    
                    # Handle Toggle Buttons (Enable/Disable Visuals)
                    elif event.ui_element in self.flag_buttons:
                        is_checked = not event.ui_element.user_data['checked']
                        event.ui_element.user_data['checked'] = is_checked
                        name = self.flag_buttons[event.ui_element]
                        
                        # Visual: [x] means ENABLED
                        mark = "[x]" if is_checked else "[ ]"
                        event.ui_element.set_text(f"{mark} {name}")

            # UI Text Updates
            self.lbl_agents.set_text(str(int(self.slider_agents.get_current_value())))
            self.lbl_k.set_text(f"{self.slider_k.get_current_value():.1f}")
            self.lbl_xi.set_text(f"{self.slider_xi.get_current_value():.2f}")
            self.lbl_speed.set_text(str(int(self.slider_speed.get_current_value())))
            self.target_sim_fps = self.slider_speed.get_current_value()

            # Simulation Logic
            if self.sim and not self.sim.is_completed():
                should_update = False
                
                if self.step_once:
                    should_update = True
                    self.step_once = False
                elif not self.paused:
                    self.sim_accumulated_time += time_delta
                    step_interval = 1.0 / self.target_sim_fps
                    if self.sim_accumulated_time >= step_interval:
                        should_update = True
                        self.sim_accumulated_time -= step_interval
                        if self.sim_accumulated_time > 1.0: self.sim_accumulated_time = 0

                if should_update:
                    self.sim.step()
                    self.update_stats()
                    if self.sim.is_completed():
                        self.lbl_status.set_text("FINISHED")
                        self.paused = True

            # Rendering
            self.screen.fill(COLOR_BG)
            self.screen.blit(self.background_surface, (self.map_offset_x, self.map_offset_y))
            
            if self.sim:
                for agent in self.sim.agentmap:
                    state = agent.state
                    screen_x = self.map_offset_x + (state.y * self.cell_size)
                    screen_y = self.map_offset_y + (state.x * self.cell_size)
                    
                    rect = (screen_x, screen_y, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, COLOR_AGENT, rect)

            self.ui_manager.update(time_delta)
            self.ui_manager.draw_ui(self.screen)
            pygame.display.flip()

if __name__ == "__main__":
    SimulationGUI().run()