# neo_model Training Dashboard

A real-time monitoring dashboard for RT-DETR model training, built with Jebi's brand identity and Material 3 Expressive design system.

---

## Overview

This dashboard provides live monitoring of:
- Training progress (epoch, batch, ETA)
- Loss metrics and AP scores
- GPU utilization (H100)
- Training configuration
- Cost tracking

**Auto-refreshes every 30 seconds** to display the latest training metrics.

---

## Design System

### Brand Integration

The dashboard implements **Jebi's brand identity** for industrial AI applications:

**Colors**:
- **Jebi Red** (#FE3B1F) - Primary accent for CTAs, progress, alerts
- **Deep Teal** (#002634) - Brand stability color in gradients
- **Dark surfaces** - Professional, reduces eye strain for 24/7 monitoring
- **High contrast** - Optimized for industrial environments

**Typography**:
- **Montserrat** (Bold, 700-800) - Headlines, card titles, branding
- **Poppins** (Regular-SemiBold, 400-600) - Body text, metrics, data

**Design Principles**:
- Material 3 Expressive design system
- 8dp spacing grid for consistency
- Smooth animations with physics-based easing
- WCAG 2.1 AA accessibility compliance
- Mobile-first responsive design

---

## File Structure

```
dashboard/
├── templates/
│   ├── index.html              # Main dashboard (Jebi-branded)
│   ├── index.html.backup       # Original version backup
│   └── index_jebi.html         # Jebi version (identical to index.html)
├── app.py                      # Flask backend server
├── DESIGN_SPECS.md             # Complete design specifications
└── README.md                   # This file
```

---

## Running the Dashboard

### Prerequisites
- Python 3.8+
- Flask

### Installation
```bash
cd dashboard
pip install flask
```

### Start Server
```bash
python app.py
```

The dashboard will be available at: `http://localhost:5000`

---

## API Endpoint

The dashboard fetches data from `/api/status`, which returns:

```json
{
  "training": {
    "epoch": 12,
    "total_epochs": 72,
    "epoch_progress": 45,
    "current_batch": 1500,
    "total_batches": 3322,
    "loss": 2.345,
    "ap": 0.523,
    "ap50": 0.685,
    "ap75": 0.562,
    "best_ap": 0.534,
    "learning_rate": 0.0001,
    "speed": 3.2,
    "time_per_epoch": 1800
  },
  "gpu": {
    "utilization": 92,
    "memory_used": 78.5,
    "temperature": 72,
    "power_draw": 450
  },
  "cost": {
    "uptime_hours": 24.5,
    "cost_so_far": 61.00,
    "projected_total": 179.28
  },
  "completion": {
    "eta": "2026-02-02 14:30",
    "hours_remaining": 48.2
  },
  "last_update_ago": "5 seconds ago"
}
```

---

## Key Features

### 1. Real-Time Progress Tracking
- Visual progress bar with gradient fill
- Current epoch and batch display
- Estimated completion time
- Hours remaining calculation

### 2. Training Metrics Display
- **Current Loss** - Large, prominent display with trend indicators
- **Average Precision (AP)** - Overall model performance
- **AP @ IoU=0.50** - Detection accuracy at 50% overlap
- **AP @ IoU=0.75** - Detection accuracy at 75% overlap
- **Best AP** - Highest AP achieved during training

### 3. GPU Monitoring
- Utilization percentage (color-coded: >85% green)
- Memory usage in GB
- Temperature in Celsius (color-coded: >80° warning)
- Power draw in Watts

### 4. Training Configuration
- Batch size with gradient accumulation
- Current learning rate (scientific notation)
- Training speed (iterations/second)
- Time per epoch (minutes)

### 5. Cost Tracking
- Instance uptime hours
- Cost accumulated so far
- Projected total cost
- Hourly rate display ($2.49/hour for H100)

### 6. Trend Indicators
Loss values show trend arrows:
- **↓ Green** - Loss decreasing (good)
- **↑ Red** - Loss increasing (concerning)
- **→ Amber** - Loss stable

---

## Design Features

### Visual Hierarchy
1. **Header** - Brand identity, status indicator, instance info
2. **Training Progress** - Most prominent card at top
3. **Metrics** - Large loss value as focal point
4. **GPU Status** - Grid layout for quick scanning
5. **Configuration** - Reference information
6. **Cost** - Business metrics
7. **Footer** - Metadata and branding

### Responsive Design

**Mobile (< 600px)**:
- Single column layout
- Smaller typography (24px titles, 36px large metrics)
- Stacked header elements
- Full-width cards with 16px padding

**Tablet (600-839px)**:
- 2-column grid for GPU metrics
- 24px body padding
- 32px page titles
- Maintained card hover effects

**Desktop (840px+)**:
- Max container width 1200px
- 32px body padding
- Full typography scale (48px large metrics)
- Enhanced hover interactions

### Accessibility

**WCAG 2.1 AA Compliance**:
- Text contrast ratios: 4.5:1 minimum (7:1 for critical data)
- Focus indicators: 3px solid Jebi Red outline
- Reduced motion support via `prefers-reduced-motion`
- Semantic HTML structure

**Keyboard Navigation**:
- Logical tab order
- Skip links for screen readers
- ARIA labels where needed

### Performance

**Optimizations**:
- CSS custom properties for tokens (no inline styles)
- Hardware-accelerated transforms
- Minimal repaints (transform vs position)
- Debounced auto-refresh (30s interval)

**Loading States**:
- Animated spinner during initial load
- Graceful error handling with styled error cards
- Smooth transitions between states

---

## Customization

### Changing Colors

Edit CSS custom properties in `<style>` section:

```css
:root {
    --jebi-red: #FE3B1F;           /* Primary accent */
    --jebi-dark: #002634;          /* Brand color */
    --success: #34D399;            /* Good metrics */
    --warning: #FEB91F;            /* Caution */
    --error: #EF4444;              /* Critical */
}
```

### Adjusting Refresh Rate

Change interval in JavaScript (30000ms = 30 seconds):

```javascript
setInterval(updateDashboard, 30000); // Change to desired ms
```

### Modifying Layout

Cards are rendered dynamically. Edit HTML templates in `updateDashboard()` function:

```javascript
html += `
    <div class="card">
        <h2>Your Card Title</h2>
        <!-- Your content -->
    </div>
`;
```

---

## Design Rationale

### Why Dark Theme?
- **Industrial context**: 24/7 monitoring in control rooms
- **Eye strain reduction**: Lower brightness for extended viewing
- **Brand alignment**: Deep teal (#002634) is core Jebi identity
- **Data focus**: Light text on dark emphasizes metrics
- **Modern aesthetic**: Professional, technical appearance

### Why Jebi Red Accents?
- **Brand recognition**: Immediate visual connection to Jebi
- **Safety critical**: Red naturally draws attention to important data
- **High contrast**: Works excellently against dark backgrounds
- **Energy**: Communicates urgency appropriate for real-time monitoring

### Why Material 3?
- **Physics-based motion**: Natural, delightful interactions
- **Adaptive design**: Seamlessly scales across all devices
- **Accessibility**: WCAG compliance built into the system
- **HCT color system**: Ensures consistent, accessible contrast

### Why These Fonts?
- **Poppins**: Clean, geometric, excellent for data display
- **Montserrat**: Strong, industrial feel for headlines
- **Web fonts**: Consistent cross-platform rendering
- **Variable weights**: Establishes clear visual hierarchy

---

## Comparison to Original

### Before (Original Design)
- Generic blue/green color scheme
- System fonts (BlinkMacSystemFont, Segoe UI)
- Basic card shadows
- Limited brand identity
- Functional but generic appearance

### After (Jebi-Branded Design)
- **Jebi Red (#FE3B1F)** primary accent
- **Deep Teal (#002634)** gradient backgrounds
- **Montserrat + Poppins** brand typography
- Material 3 elevation system
- Strong brand presence throughout
- Professional, industrial aesthetic
- Enhanced visual hierarchy
- Improved accessibility (WCAG 2.1 AA)
- Smoother animations (physics-based easing)
- Better mobile responsiveness

---

## Browser Support

**Tested and supported**:
- Chrome 90+ (recommended)
- Firefox 88+
- Safari 14+
- Edge 90+

**Required features**:
- CSS Custom Properties
- CSS Grid
- Flexbox
- Fetch API
- ES6 JavaScript

---

## Future Enhancements

### Phase 2 Features
- [ ] Light/Dark theme toggle
- [ ] Chart visualizations (Chart.js integration)
  - Loss curve over epochs
  - AP progression chart
  - GPU utilization timeline
- [ ] Browser notifications for critical events
- [ ] Historical data view (past training runs)
- [ ] Export functionality (PDF reports)
- [ ] WebSocket support (real-time push updates)

### Component Additions
- [ ] Status badges (running, paused, completed, failed)
- [ ] Action buttons (pause, resume, stop training)
- [ ] Dropdown filters (epoch range, metric selection)
- [ ] Tab navigation (metrics, logs, configuration)
- [ ] Side panel for detailed logs

---

## Contributing

When making changes:

1. **Follow design system** - Use CSS custom properties from `:root`
2. **Maintain accessibility** - Test with screen readers, keyboard navigation
3. **Test responsive** - Verify on mobile (< 600px), tablet, desktop
4. **Check contrast** - Ensure WCAG 2.1 AA compliance (4.5:1 minimum)
5. **Document changes** - Update this README and DESIGN_SPECS.md

---

## Resources

### Design System
- [DESIGN_SPECS.md](/dashboard/DESIGN_SPECS.md) - Complete specifications
- [Material 3 Guidelines](https://m3.material.io/) - Official documentation
- [Jebi Brand Assets](https://drive.google.com/drive/folders/1prLCC96yxWSHLiNndw8svI_PajwjaNGN) - Logo, colors, guidelines

### Typography
- [Montserrat Font](https://fonts.google.com/specimen/Montserrat)
- [Poppins Font](https://fonts.google.com/specimen/Poppins)

### Accessibility
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)

---

## License

Part of the neo_model RT-DETR training project.

**Design System**: Jebi Brand Identity v2.0
**Framework**: Material 3 Expressive
**Last Updated**: January 31, 2026
