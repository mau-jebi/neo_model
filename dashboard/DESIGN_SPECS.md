# neo_model Training Dashboard - Jebi Design Specifications

## Design System Overview

This dashboard implements **Material 3 Expressive Design System** integrated with **Jebi Brand Identity** for industrial AI applications.

---

## 1. Color Palette (HCT System)

### Primary Colors
```css
--jebi-red: #FE3B1F              /* Primary accent, CTAs, alerts */
--jebi-red-bright: #FE1E1F       /* Hover states, emphasis */
--jebi-dark: #002634             /* Brand color, stability */
--jebi-blue-medium: #174F64      /* Secondary actions */
--jebi-blue-light: #5396A5       /* Tertiary elements */
--jebi-gray-light: #C3C6C8       /* Borders, subtle accents */
```

### Surface Colors (Dark Theme)
```css
--surface: #0A1419               /* Base background */
--surface-dim: #000A0F           /* Deepest surface */
--surface-bright: #1A2429        /* Elevated surface */
--surface-container: #162126     /* Card background */
--surface-container-high: #1D282D     /* Nested elements */
--surface-container-highest: #283338  /* Top layer */
--on-surface: #E1E3E5            /* Primary text */
--on-surface-variant: #BFC8CC    /* Secondary text */
```

### Semantic Colors
```css
--success: #34D399    /* Good metrics, completion */
--warning: #FEB91F    /* Caution indicators */
--error: #EF4444      /* Critical alerts */
```

### Contrast Requirements
- Normal text on background: **4.5:1 minimum** (WCAG AA)
- Large text (18px+): **3:1 minimum**
- UI components: **3:1 minimum**
- Critical alerts: **7:1 target** (WCAG AAA)

---

## 2. Typography

### Font Families
```css
--font-headline: 'Montserrat', sans-serif;  /* Headlines, branding */
--font-body: 'Poppins', sans-serif;         /* Body text, data */
```

### Type Scale
```
Display Large:    48px / 700 / Poppins      (Large metrics)
Headline Large:   32px / 700 / Montserrat   (Page title)
Headline Small:   18px / 700 / Montserrat   (Card titles)
Body Large:       16px / 500 / Poppins      (Body text)
Label Large:      14px / 600 / Poppins      (Metric labels)
Label Small:      12px / 600 / Poppins      (Grid labels)
```

### Typography Usage
- **Card Titles**: 18px Montserrat Bold, uppercase, 0.8px letter-spacing, Jebi Red
- **Metric Labels**: 14px Poppins Medium, uppercase, 0.4px letter-spacing, variant color
- **Metric Values**: 20px Poppins SemiBold, on-surface color
- **Large Values**: 48px Poppins Bold, Jebi Red, -1.5px letter-spacing

---

## 3. Spacing System (8dp Grid)

### Spacing Tokens
```css
--spacing-4: 4px      /* Tight spacing */
--spacing-8: 8px      /* Component internal */
--spacing-12: 12px    /* Related elements */
--spacing-16: 16px    /* Mobile margins */
--spacing-20: 20px    /* Standard spacing */
--spacing-24: 24px    /* Card padding */
--spacing-32: 32px    /* Desktop margins */
--spacing-40: 40px    /* Section spacing */
```

### Layout Grid
- **Mobile (< 600px)**: 16px margins, single column
- **Tablet (600-839px)**: 24px margins, 2-column grid
- **Desktop (840px+)**: 32px margins, max-width 1200px

---

## 4. Shape System (Border Radius)

```css
--corner-small: 8px       /* Grid metrics, nested cards */
--corner-medium: 12px     /* Main cards, dialogs */
--corner-large: 16px      /* Header, large containers */
--corner-full: 999px      /* Pills, progress bars */
```

### Component Shapes
- **Header**: 16px (large)
- **Cards**: 12px (medium)
- **Grid Metrics**: 8px (small)
- **Progress Bar**: 999px (full pill)
- **Badges**: 999px (full pill)

---

## 5. Elevation System

```css
--elevation-1: 0 1px 3px rgba(0, 0, 0, 0.3), 0 1px 2px rgba(0, 0, 0, 0.24);
--elevation-2: 0 3px 6px rgba(0, 0, 0, 0.4), 0 2px 4px rgba(0, 0, 0, 0.3);
--elevation-3: 0 10px 20px rgba(0, 0, 0, 0.5), 0 3px 6px rgba(0, 0, 0, 0.4);
```

### Component Elevations
- **Cards (resting)**: Elevation 1
- **Cards (hover)**: Elevation 2 + translateY(-2px)
- **Header**: Elevation 1
- **Modals/Dialogs**: Elevation 3

---

## 6. Motion System

### Duration Tokens
```css
--duration-fast: 200ms       /* Small components (buttons) */
--duration-default: 350ms    /* Standard transitions */
--duration-slow: 500ms       /* Complex animations */
```

### Easing Functions
```css
--easing-standard: cubic-bezier(0.2, 0, 0, 1);           /* Standard transitions */
--easing-emphasized: cubic-bezier(0.05, 0.7, 0.1, 1);    /* Emphasized motion */
```

### Animation Patterns
- **Card Hover**: Transform Y + elevation change (350ms)
- **Progress Bar**: Width transition (500ms emphasized easing)
- **Status Indicator**: Pulse animation (2s infinite)
- **Loading Spinner**: Rotate animation (1s linear infinite)

### Accessibility
```css
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
```

---

## 7. Component Specifications

### Header Component
- **Background**: surface-container
- **Padding**: 24px
- **Border**: 1px solid rgba(254, 59, 31, 0.15)
- **Border Radius**: 16px (large)
- **Shadow**: Elevation 1
- **Gradient Accent**: Right-side subtle red gradient overlay

**Elements**:
- Status Indicator: 12px red circle, pulsing animation
- Title: 32px Montserrat Bold, "neo_model" in Jebi Red
- Instance Badge: Pill-shaped, 14px Poppins Medium

### Card Component
- **Background**: surface-container
- **Padding**: 24px
- **Border**: 1px solid rgba(255, 255, 255, 0.06)
- **Border Radius**: 12px (medium)
- **Shadow**: Elevation 1 (hover: Elevation 2)
- **Margin Bottom**: 20px

**Title Style**:
- Font: 18px Montserrat Bold
- Color: Jebi Red (#FE3B1F)
- Text Transform: Uppercase
- Letter Spacing: 0.8px
- Border Bottom: 3px solid rgba(254, 59, 31, 0.4)
- Padding Bottom: 12px

### Progress Bar Component
- **Height**: 40px
- **Background**: surface-container-high
- **Border**: 2px solid rgba(254, 59, 31, 0.3)
- **Border Radius**: 999px (full pill)
- **Fill**: Linear gradient (Jebi Red to Bright Red)
- **Fill Overlay**: White gradient (top to bottom, 15% opacity)
- **Transition**: Width 500ms emphasized easing
- **Text**: 16px Poppins Bold, white

### Metric Row Component
- **Padding**: 16px vertical
- **Border Bottom**: 1px solid rgba(255, 255, 255, 0.04)
- **Layout**: Flexbox space-between

**Label**:
- Font: 14px Poppins Medium
- Color: on-surface-variant
- Text Transform: Uppercase
- Letter Spacing: 0.4px

**Value**:
- Font: 20px Poppins SemiBold
- Color: on-surface (default)
- Large variant: 48px, Jebi Red

### Grid Metric Component
- **Background**: surface-container-high
- **Padding**: 20px
- **Border**: 1px solid rgba(255, 255, 255, 0.04)
- **Border Radius**: 8px (small)
- **Hover**: Transform Y(-2px), border color Jebi Red 30%

**Label**:
- Font: 12px Poppins SemiBold
- Color: on-surface-variant
- Text Transform: Uppercase
- Letter Spacing: 0.6px

**Value**:
- Font: 28px Poppins Bold
- Color: on-surface

---

## 8. State Indicators

### Color-Coded States
- **Good/Success**: #34D399 (green) - High GPU utilization, low loss
- **Warning**: #FEB91F (amber) - High temperature, projected costs
- **Error/Critical**: #EF4444 (red) - Critical alerts, failures
- **Neutral**: on-surface - Default metrics

### Trend Indicators
- **Decreasing (good for loss)**: ↓ green arrow
- **Increasing (bad for loss)**: ↑ red arrow
- **Stable**: → amber arrow

---

## 9. Responsive Breakpoints

### Mobile (< 600px)
- Body padding: 16px
- Single column grid
- Title: 24px
- Large metrics: 36px
- Card padding: 16px

### Tablet (600-839px)
- Body padding: 24px
- 2-column grid for metrics
- Title: 32px
- Large metrics: 48px
- Card padding: 24px

### Desktop (840px+)
- Body padding: 32px
- Max container width: 1200px
- 2-column grid for metrics
- Full typography scale

---

## 10. Accessibility Requirements

### Contrast Ratios
All text and UI elements meet WCAG 2.1 AA standards:
- Small text: 4.5:1 minimum
- Large text: 3:1 minimum
- UI components: 3:1 minimum

### Keyboard Navigation
- All interactive elements accessible via Tab
- Focus indicators: 3px solid Jebi Red, 2px offset
- Logical tab order (top-to-bottom, left-to-right)

### Motion Sensitivity
- Respects `prefers-reduced-motion` media query
- All animations disabled for reduced motion users

### Screen Reader Support
- Semantic HTML structure
- ARIA labels for icon-only elements
- Status updates announced via live regions

---

## 11. Brand Integration

### Jebi Identity Elements
- Primary accent: Jebi Red (#FE3B1F) for emphasis
- Stability color: Deep Teal (#002634) in gradients
- Industrial aesthetic: High contrast, bold typography
- Professional feel: Clean layouts, subtle animations

### Brand Applications
- **Headers**: Red accent borders and status indicators
- **Progress**: Red gradient fills for active state
- **Typography**: Bold Montserrat for headlines (industrial strength)
- **Spacing**: Generous breathing room (professional clarity)

---

## 12. Implementation Notes

### HTML Structure
```html
<header>
  <div class="header-content">
    <div class="header-left">
      <span class="status-indicator"></span>
      <h1><span class="logo-accent">neo_model</span> Training Monitor</h1>
    </div>
    <div class="instance-badge">H100 • 209.20.157.204</div>
  </div>
</header>

<div class="card">
  <h2>Card Title</h2>
  <div class="metric-row">
    <span class="metric-label">Label</span>
    <span class="metric-value">Value</span>
  </div>
</div>
```

### CSS Custom Properties
All design tokens are defined as CSS custom properties in `:root` for easy theming and maintenance.

### JavaScript Behavior
- Auto-refresh every 30 seconds
- Dynamic trend calculation (compare to previous value)
- Responsive error handling with styled error cards
- Loading state with animated spinner

---

## 13. File Structure

```
dashboard/
├── templates/
│   └── index.html          # Complete dashboard (HTML + CSS + JS)
├── DESIGN_SPECS.md         # This file
└── app.py                  # Flask backend
```

---

## 14. Design Rationale

### Why Dark Theme?
- **Industrial context**: Reduces eye strain in 24/7 monitoring
- **Brand alignment**: Deep teal (#002634) is core Jebi identity
- **Data focus**: Light text on dark draws attention to metrics
- **Modern aesthetic**: Professional, technical appearance

### Why Material 3 Expressive?
- **Physics-based motion**: Natural, delightful interactions
- **Adaptive components**: Scales seamlessly across devices
- **Accessibility built-in**: WCAG compliance by design
- **HCT color system**: Ensures consistent contrast ratios

### Why Jebi Red Accents?
- **Brand recognition**: Immediate Jebi identity
- **Safety critical**: Red draws attention to important data
- **High contrast**: Works well against dark backgrounds
- **Energy and urgency**: Appropriate for real-time monitoring

---

## 15. Future Enhancements

### Phase 2 Features
- [ ] Dark/Light theme toggle
- [ ] Chart visualizations (loss curves, AP progression)
- [ ] Alert notifications (browser notifications for critical events)
- [ ] Historical data view (past training runs)
- [ ] Export functionality (PDF reports)

### Component Library Expansion
- [ ] Status badges (running, paused, completed)
- [ ] Action buttons (pause, resume, stop)
- [ ] Dropdown filters (epoch range, metric type)
- [ ] Tabs (metrics, logs, configuration)

---

**Design System Version**: 1.0
**Last Updated**: January 31, 2026
**Designer**: Jebi Design Team
**Framework**: Material 3 Expressive + Jebi Brand Identity
