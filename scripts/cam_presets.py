"""
Module: scripts.cam_presets
"""

CAMERA_PRESETS = {

    # ── YOUR ORIGINAL 8 ────────────────────────────────────────────────────

    "hero": {
        "cam_pos": [4.5, 2.5, 10.0], "look_at": [0.0, 0.0, 0.0], "fov": 55,
        "note": "Interstellar-ish glamour shot. Disk visible, strong Doppler, large shadow."
    },
    "high_inclination": {
        "cam_pos": [0.0, 8.0, 12.0], "look_at": [0.0, 0.0, 0.0], "fov": 60,
        "note": "Secondary image wraps over the top. Physics over cinematics."
    },
    "edge_on": {
        "cam_pos": [0.0, 0.8, 15.0], "look_at": [0.0, 0.0, 0.0], "fov": 70,
        "note": "Razor-thin disk. Lensing lifts it above and below the shadow."
    },
    "polar": {
        "cam_pos": [0.0, 15.0, 0.0], "look_at": [0.0, 0.0, 0.0], "fov": 70,
        "note": "Down the spin axis. Circular shadow, ring disk. Symmetry test."
    },
    "near_horizon": {
        "cam_pos": [3.0, 0.4, 4.0], "look_at": [0.0, 0.0, 0.0], "fov": 90,
        "note": "Shadow fills most of frame. Disk wraps everywhere. Expensive."
    },
    "frame_dragging": {
        "cam_pos": [8.0, 0.2, 10.0], "look_at": [0.0, 0.0, 0.0], "fov": 50,
        "note": "a=0.998 showcase. Prograde brightening, shadow asymmetry visible."
    },
    "wallpaper": {
        "cam_pos": [6.0, 3.5, 9.0], "look_at": [0.0, 0.0, 0.0], "fov": 45,
        "note": "Narrow FOV, large shadow, clean disk wrap. High keepability."
    },
    "doctor_was_worth_it": {
        "cam_pos": [2.5, 1.0, 6.0], "look_at": [0.0, 0.0, 0.0], "fov": 110,
        "note": "Ultra-wide close-up. Aggressive lensing. Bug revealer."
    },
    "disk_texture": {
    "cam_pos": [0.0, 12.0, 6.0], "look_at": [0.0, 0.0, 0.0], "fov": 55,
    "note": "High inclination close-up. Disk fills most of frame. "
            "Best preset to evaluate spiral arm structure and density "
            "variation after the disk_density port."
},

    # ── NEW: SHADOW GEOMETRY ───────────────────────────────────────────────

    "d_shadow": {
        # The D-shaped shadow is most visible from near-equatorial with
        # a moderate FOV that fits the full shadow without too much disk.
        # Prograde side (left at this angle) has smaller critical impact
        # parameter — shadow squashes on that side, producing the D-shape.
        "cam_pos": [0.0, 0.3, 18.0], "look_at": [0.0, 0.0, 0.0], "fov": 30,
        "note": "Tight FOV, near-equatorial. D-shaped shadow from spin asymmetry. "
                "Best angle to see prograde/retrograde capture radius difference."
    },
    "shadow_edge_zoom": {
        # Very tight FOV zoomed into the prograde shadow edge.
        # This is where you'd resolve the photon ring if resolution is high enough.
        # At 1920x1080 and FOV=8 the shadow edge subtends ~30 pixels.
        "cam_pos": [0.0, 0.3, 18.0], "look_at": [-0.8, 0.0, 0.0], "fov": 8,
        "note": "Extreme zoom into prograde shadow edge. Photon ring may resolve "
                "at 1920x1080. Expensive — many near-critical rays in frame."
    },

    # ── NEW: LENSING STRUCTURE ─────────────────────────────────────────────

    "lensing_showcase": {
        # Positioned to show primary disk image below and lensed ghost above
        # simultaneously, with the shadow clearly separating them.
        "cam_pos": [0.0, 4.0, 14.0], "look_at": [0.0, -0.5, 0.0], "fov": 65,
        "note": "Look-at offset downward to frame primary disk lower, ghost upper. "
                "Secondary image should appear as thin arc above shadow."
    },
    "behind_the_disk": {
        # Camera slightly below the disk plane looking up through it.
        # Disk is transparent in thin regions — produces a completely
        # different lensing topology from the usual above-plane shots.
        "cam_pos": [0.0, -2.0, 14.0], "look_at": [0.0, 0.5, 0.0], "fov": 60,
        "note": "Camera below equatorial plane looking up. Inverts the usual "
                "lensing geometry — disk curves away, shadow above field center."
    },
    "retrograde_side": {
        # Positioned so the retrograde (dim, redshifted) side faces camera.
        # Most renders show the prograde bright side prominently — this
        # isolates the opposite hemisphere for comparison.
        "cam_pos": [-8.0, 0.3, 10.0], "look_at": [0.0, 0.0, 0.0], "fov": 55,
        "note": "Retrograde hemisphere prominent. Disk should look dimmer and redder "
                "than prograde shots. Good Doppler asymmetry comparison."
    },
    "gravitational_lens": {
        "cam_pos": [0.0, 0.1, 30.0], "look_at": [0.0, 0.0, 0.0], "fov": 8,
        "note": "Very distant, very tight. BH acts as pure gravitational lens. "
            "Multiple images of background stars visible around shadow edge."
    },

    # ── NEW: CINEMATIC / FLYBY ─────────────────────────────────────────────

    "flyby_close": {
        # Classic cinematic angle — camera slightly above equatorial, offset
        # to one side, looking slightly past the BH rather than directly at it.
        # The disk sweeps dramatically across the lower frame.
        "cam_pos": [5.0, 1.5, 8.0], "look_at": [-2.0, 0.0, 0.0], "fov": 75,
        "note": "Cinematic flyby. Look-at offset left so BH isn't dead-center. "
                "Disk dominates lower frame. Natural composition."
    },
    "dutch_angle": {
        # Same position as hero but with roll applied.
        # Tests that roll doesn't introduce artifacts in the lensing.
        "cam_pos": [4.5, 2.5, 10.0], "look_at": [0.0, 0.0, 0.0], "fov": 55,
        "roll": 25.0,
        "note": "Hero position with 25-degree roll. Cinematic tension. "
                "Also validates roll doesn't corrupt lensing geometry."
    },
    "over_the_shoulder": {
        # Camera nearly behind the BH — the disk is lensed into a full
        # Einstein ring rather than the usual one-sided arc.
        "cam_pos": [0.5, 0.3, -18.0], "look_at": [0.0, 0.0, 0.0], "fov": 40,
        "note": "Camera on far side of BH. Einstein ring / full disk wrap. "
                "Prograde/retrograde completely swapped vs standard shots."
    },
    "orbital_mechanics": {
        # Elevated at 45 degrees, offset to show the disk as an oval.
        # Looks like you're in a spacecraft in a higher orbit watching.
        "cam_pos": [8.0, 8.0, 8.0], "look_at": [0.0, 0.0, 0.0], "fov": 50,
        "note": "45-degree elevation. Disk appears as oval. Spacecraft perspective."
    },

    # ── NEW: RESEARCH / COMPARISON ─────────────────────────────────────────

    "schwarzschild_equivalent": {
        # Same as hero but designed to be run at a=0 for direct comparison.
        # Circular shadow, symmetric disk — the baseline before spin.
        "cam_pos": [4.5, 2.5, 10.0], "look_at": [0.0, 0.0, 0.0], "fov": 55,
        "note": "Same as hero. Run with --spin 0 for Schwarzschild comparison. "
                "Shadow should be circular and disk symmetric."
    },
    "isco_comparison": {
        # Very close to BH, tight FOV, framed to show the inner disk edge.
        # At a=0.998 ISCO is at ~1.063 — much closer than Schwarzschild's 6M.
        # Run at a=0 and a=0.998 to see ISCO shrinkage directly.
        "cam_pos": [2.0, 0.3, 5.0], "look_at": [0.0, 0.0, 0.0], "fov": 40,
        "note": "Tight FOV on inner disk edge. Run at a=0 and a=0.998 — ISCO "
                "moves from r=6 to r≈1.06. Inner disk completely different."
    },
    "luminet_1979": {
        # Approximating the camera setup from Luminet's 1979 paper.
        # Observer at 50M distance, 10 degrees above equatorial plane.
        # The closest thing to the historically correct reference view.
        "cam_pos": [0.0, 8.7, 49.2], "look_at": [0.0, 0.0, 0.0], "fov": 20,
        "note": "Approximate Luminet (1979) setup. Observer at 50M, ~10° inclination. "
                "Historical reference — compare your output against his Figure 3."
    },
    "polar_near": {
        # Same axis as polar but much closer — shadow fills more of frame.
        # Tests polar axis stability at short distance.
        "cam_pos": [0.0, 6.0, 0.0], "look_at": [0.0, 0.0, 0.0], "fov": 80,
        "note": "Close polar view. Shadow large. Tests polar axis numerics "
                "at closer range than the standard polar preset."
    },
    "eht_analog": {
        # M87*-inspired — nearly face-on with a slight inclination.
        # EHT images M87* at ~17° inclination from the jet axis.
        "cam_pos": [0.0, 5.1, 16.5], "look_at": [0.0, 0.0, 0.0], "fov": 25,
        "note": "M87*-analog. ~17° from spin axis, tight FOV. "
                "Compare shadow shape to EHT 2019 image qualitatively."
    },
    "critical_curve": {
        # Extreme zoom on the critical curve. Shadow edge should be ~1.5M in radius.
        # Run at high resolution to resolve the photon ring.
        "cam_pos": [0.0, 0.2, 25.0], "look_at": [0.0, 0.0, 0.0], "fov": 6,
        "note": "Extreme zoom on the critical curve. Shadow edge should be ~1.5M "
                "in radius. Run at high resolution to resolve the photon ring."
    },
    "extreme_close": {
        # As close as you can get without the shadow eating the whole frame.
        # Produces the most extreme lensing visible in a single render.
        "cam_pos": [1.5, 0.2, 3.0], "look_at": [0.0, 0.0, 0.0], "fov": 120,
        "note": "Extreme proximity, ultra-wide. Shadow ~60 percent of frame. "
                "Maximum lensing distortion. Very expensive — most pixels "
                "trace near-critical geodesics."
    },
    "penrose_zone": {
    "cam_pos": [0.0, 0.5, 6.0], "look_at": [0.0, 0.0, 0.0], "fov": 35,
    "note": "Tight framing on the ergosphere region at a=0.998. "
            "The asymmetric brightness between prograde and retrograde "
            "orbits is most visible at this distance and FOV."
},
    "photon_ring": {
    "cam_pos": [0.0, 0.1, 20.0], "look_at": [0.0, 0.0, 0.0], "fov": 12,
    "note": "Near-equatorial, tight FOV centered on shadow boundary. "
            "At 1920x1080 the secondary photon ring should be resolvable "
            "as a distinct bright arc just outside the shadow."
},
}