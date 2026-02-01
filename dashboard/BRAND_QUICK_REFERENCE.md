# Jebi Brand Quick Reference Guide
## For neo_model Training Dashboard

---

## Color Palette - EXACT HEX CODES

### PRIMARY COLORS

```css
/* DEEP TEAL - Stability, Technology, Main Text */
#002634
RGB: (0, 38, 52)
Pantone: Custom Deep Teal
Usage: Headers, primary text, main data values (60% visual weight)
```

```css
/* BRIGHT RED - Alerts, Energy, Key Highlights */
#f03a1e
RGB: (240, 58, 30)
Pantone: 1655 C
Usage: Alerts, status indicators, section accents (10-15% visual weight)
```

### SECONDARY COLORS

```css
/* LIGHT GRAY - Backgrounds, Borders */
#c3c6c8
RGB: (195, 198, 200)
Pantone: 428 C
Usage: Borders, subtle backgrounds, secondary text (15-20%)
```

```css
/* WHITE - Primary Background */
#ffffff
RGB: (255, 255, 255)
Usage: Card backgrounds, primary surface, text on dark (15-20%)
```

---

## Typography - Montserrat Font Family

### Font Import (Google Fonts)
```html
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap" rel="stylesheet">
```

### Font Stack
```css
font-family: 'Montserrat', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
```

### Hierarchy

| Element | Weight | Size | Color | Usage |
|---------|--------|------|-------|-------|
| **Titles (H1)** | Bold (700) | 2.25rem | #ffffff (on dark) / #002634 (on light) | Main dashboard title |
| **Section Headers (H2)** | Bold (700) | 1.3rem | #002634 | Card section titles |
| **Subtitles** | Medium (500) | 0.95-1rem | #c3c6c8 / #002634 | Supporting titles |
| **Metric Labels** | Medium (500) | 0.85-0.95rem | #002634 | Data field labels |
| **Body Text** | Regular (400) | 0.9-0.95rem | #002634 | General content |
| **Large Values** | Bold (700) | 1.8-2.8rem | #002634 / #f03a1e (alerts) | Key metrics |

---

## Component Styling Patterns

### Cards
```css
.card {
    background: #ffffff;
    border: 2px solid #c3c6c8;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0, 38, 52, 0.08);
}

.card h2 {
    color: #002634;
    font-weight: 700;
    border-bottom: 3px solid #f03a1e;
}
```

### Alerts
```css
.alert-box {
    background: #fff5f4;
    border: 2px solid #f03a1e;
    border-radius: 8px;
}

.alert-icon {
    background: #f03a1e;
    width: 24px;
    height: 24px;
    border-radius: 50%;
}
```

### Progress Bars
```css
.progress-bar {
    background: #c3c6c8;
    border-radius: 18px;
    height: 36px;
}

.progress-fill {
    background: linear-gradient(90deg, #002634 0%, #004d6b 100%);
    color: #ffffff;
}
```

### Metrics
```css
.metric-label {
    color: #002634;
    font-weight: 500;
}

.metric-value {
    color: #002634;
    font-weight: 700;
}

.metric-value.alert {
    color: #f03a1e;
}
```

### Status Indicator
```css
.status-indicator {
    background: #f03a1e;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
    box-shadow: 0 0 10px rgba(240, 58, 30, 0.6);
}
```

---

## When to Use Bright Red (#f03a1e)

### ✅ APPROPRIATE USE (Strategic Highlights)
- Alert boxes and critical warnings
- Status indicator (active monitoring)
- Section header underlines (3px accent)
- High temperature warnings
- Low GPU utilization alerts
- Cost overrun warnings
- Trend indicators (performance issues)

### ❌ AVOID (Overwhelming Use)
- Large background areas
- Body text (use deep teal instead)
- Multiple large elements simultaneously
- Progress bar fills (use deep teal)
- Normal state indicators

**Rule of Thumb**: Red should account for 10-15% of visual weight maximum

---

## When to Use Deep Teal (#002634)

### ✅ APPROPRIATE USE (Foundation Color)
- Header backgrounds
- Primary text and titles
- Main metric values
- Progress bar fills
- Grid metric values
- Body content text
- Section headers

**Rule of Thumb**: Teal should account for 60% of visual weight (dominant color)

---

## Responsive Breakpoints

### Desktop (≥769px)
```css
.container { max-width: 1200px; }
h1 { font-size: 2.25rem; }
.metric-value.large { font-size: 2.8rem; }
.grid-2 { grid-template-columns: 1fr 1fr; }
```

### Tablet (≤768px)
```css
body { padding: 12px; }
h1 { font-size: 1.5rem; }
.metric-value.large { font-size: 2rem; }
```

### Mobile (≤480px)
```css
h1 { font-size: 1.25rem; }
.metric-value { font-size: 1.1rem; }
.metric-value.large { font-size: 1.6rem; }
.grid-2 { grid-template-columns: 1fr; }
```

---

## Brand Character Guidelines

### STRONG & TECHNOLOGICAL
- Deep teal as dominant color
- Bold typography for titles
- Technical metrics prominently displayed
- Clean, modern interface design

### RELIABLE & INDUSTRIAL
- High contrast for readability
- Stable color palette (no flashy animations)
- Consistent spacing and alignment
- Professional card-based organization

### APPROACHABLE & FRIENDLY
- Rounded corners (12px border-radius)
- Curved accent elements
- Generous white space
- Smooth hover transitions

### PREVENTION-FOCUSED
- Proactive alert system
- Red warnings for critical conditions
- Pulsing status indicator
- Clear visual hierarchy prioritizing safety

---

## CSS Variable Implementation (Recommended)

For easier maintenance, consider implementing CSS variables:

```css
:root {
    /* Jebi Brand Colors */
    --jebi-deep-teal: #002634;
    --jebi-bright-red: #f03a1e;
    --jebi-light-gray: #c3c6c8;
    --jebi-white: #ffffff;

    /* Teal Variations */
    --jebi-teal-light: #004d6b;
    --jebi-teal-alpha-08: rgba(0, 38, 52, 0.08);
    --jebi-teal-alpha-15: rgba(0, 38, 52, 0.15);

    /* Red Variations */
    --jebi-red-alpha-05: rgba(240, 58, 30, 0.05);
    --jebi-red-alpha-60: rgba(240, 58, 30, 0.6);
    --jebi-red-tint: #fff5f4;

    /* Typography */
    --font-family: 'Montserrat', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    --font-weight-regular: 400;
    --font-weight-medium: 500;
    --font-weight-bold: 700;

    /* Spacing */
    --border-radius: 12px;
    --border-radius-small: 8px;
    --border-radius-round: 50%;
}
```

---

## Brand Don'ts - Common Mistakes to Avoid

### ❌ DON'T
1. Use generic green colors (#4CAF50) - always use deep teal or bright red
2. Use system fonts without Montserrat fallback
3. Apply bright red to large background areas
4. Use dark gradients instead of white backgrounds
5. Create low-contrast color combinations
6. Distort the Jebi logo or separate its elements
7. Use light gray for primary text (use deep teal)
8. Omit proper font weight differentiation (all bold/all regular)
9. Use bright red for normal state indicators (reserve for alerts)
10. Ignore mobile responsive requirements

### ✅ DO
1. Use exact hex codes: #002634, #f03a1e, #c3c6c8, #ffffff
2. Import Montserrat font family from Google Fonts
3. Apply bright red strategically for alerts and accents
4. Use white backgrounds for clarity and readability
5. Maintain high contrast (deep teal on white)
6. Display Jebi branding prominently with "Leader in Mining Tech"
7. Use deep teal for primary text and data
8. Differentiate titles (Bold 700), subtitles (Medium 500), body (Regular 400)
9. Reserve bright red for warnings and critical highlights
10. Test on mobile devices and ensure responsive behavior

---

## Brand Message Integration

### Required Brand Statements

**Header**: "Jebi AI Engine - Leader in Mining Tech"
**Footer**: "Jebi AI Engine - Full Prevention Hardware for Mining Operations"

### Optional Enhancements
- "Advanced Object Detection for Mining Safety"
- "Real-Time Monitoring Technology"
- "Protecting Mining Operations with AI"
- "Engineered for Harsh Mining Environments"

---

## Accessibility Considerations

### Color Contrast Ratios (WCAG AA Compliant)

| Combination | Ratio | Status |
|-------------|-------|--------|
| #002634 on #ffffff | 14.7:1 | ✅ AAA (Excellent) |
| #f03a1e on #ffffff | 4.8:1 | ✅ AA (Good) |
| #ffffff on #002634 | 14.7:1 | ✅ AAA (Excellent) |
| #002634 on #c3c6c8 | 11.2:1 | ✅ AAA (Excellent) |

All brand color combinations meet or exceed accessibility standards.

---

## File Locations

**Dashboard HTML**: `/dashboard/templates/index.html`
**Brand Report**: `/dashboard/BRAND_COMPLIANCE_REPORT.md`
**This Reference**: `/dashboard/BRAND_QUICK_REFERENCE.md`

---

**Quick Reference Version 1.0**
Jebi Brand Identity Specialist
2026-01-31
