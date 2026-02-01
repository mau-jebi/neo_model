# JEBI BRAND STANDARDS - DEFINITIVE REFERENCE
## For Training Dashboard Implementation

---

## CRITICAL: Non-Negotiable Brand Requirements

This document defines the absolute, non-negotiable requirements for Jebi brand compliance. ANY deviation from these standards compromises brand integrity.

---

## 1. COLOR PALETTE (EXACT HEX CODES REQUIRED)

### PRIMARY COLORS

**Deep Teal (MUST USE EXACT):**
```
HEX: #002634
RGB: (0, 38, 52)
Usage: 60% of visual weight
Purpose: Stability, technology, main text, headers
```

**Bright Red (MUST USE EXACT):**
```
HEX: #f03a1e
Pantone: 1655 C
RGB: (240, 58, 30)
Usage: 10-15% of visual weight
Purpose: Alerts, energy, key highlights
```

**CRITICAL:** Do NOT use approximations like #FE3B1F, #FE1E1F, or any other red variant.

### SECONDARY COLORS

**Light Gray:**
```
HEX: #c3c6c8
Pantone: 428 C
RGB: (195, 198, 200)
Usage: 15-20% of visual weight
Purpose: Backgrounds, borders, accents
```

**White:**
```
HEX: #ffffff
RGB: (255, 255, 255)
Usage: 15-20% of visual weight
Purpose: Primary background, clean surfaces
```

### PROHIBITED COLORS

❌ DO NOT USE:
- Dark mode surfaces (#0A1419, #1A2429, etc.)
- Generic greens (#34D399, #4CAF50)
- Generic blues (except Jebi blue variants if documented)
- Material Design color tokens
- Any red other than #f03a1e

---

## 2. TYPOGRAPHY (MONTSERRAT ONLY)

### Font Family

**REQUIRED:**
```css
font-family: 'Montserrat', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
```

### Font Weights

| Weight | Usage | CSS |
|--------|-------|-----|
| Bold (700) | Titles, headers, data values | font-weight: 700; |
| Medium (500) | Subtitles, labels, alerts | font-weight: 500; |
| Regular (400) | Body text, descriptions | font-weight: 400; |

### PROHIBITED Typography

❌ DO NOT USE:
- Poppins (not part of Jebi brand)
- Sofia (reference only, not for implementation)
- Roboto, Inter, or other generic fonts
- System fonts as primary (only as fallback)
- Font weights 600 or 800 (use 500 or 700)

### Typography Hierarchy

| Element | Weight | Size (Desktop) | Color |
|---------|--------|----------------|-------|
| H1 Main Title | 700 | 2.25rem (36px) | #ffffff (on dark) / #002634 (on light) |
| H2 Section Title | 700 | 1.3rem (20.8px) | #002634 |
| Brand Title | 500 | 1rem (16px) | #c3c6c8 |
| Metric Labels | 500 | 0.95rem (15.2px) | #002634 |
| Large Values | 700 | 2.8rem (44.8px) | #002634 / #f03a1e (alerts) |
| Body Text | 400 | 0.9-0.95rem | #002634 |

---

## 3. BACKGROUND & SURFACE COLORS

### PRIMARY BACKGROUND

**REQUIRED:**
```css
body {
    background: #ffffff;
    color: #002634;
}
```

### CARD BACKGROUNDS

**REQUIRED:**
```css
.card {
    background: #ffffff;
    border: 2px solid #c3c6c8;
}
```

### PROHIBITED Backgrounds

❌ DO NOT USE:
- Dark backgrounds (#0A1419, #1A2429, etc.)
- Dark gradients
- Material Design surface colors
- Transparent/glassmorphism effects with dark themes

**REASON:** Jebi dashboards must be readable in bright mining environments with high ambient light. Dark themes reduce visibility and professionalism.

---

## 4. DESIGN SYSTEM

### REQUIRED APPROACH

**Jebi Brand Identity System**
- Not Material Design
- Not Material 3 Expressive
- Not Fluent Design
- Not any third-party design system

**Core Principles:**
1. Strong, technological, reliable
2. Industrial-grade professionalism
3. High contrast for harsh environments
4. Curved elements for approachability
5. Prevention-focused (alerts and warnings)

### PROHIBITED Design Systems

❌ DO NOT implement:
- Material Design tokens or components
- Material 3 Expressive
- Bootstrap default styling
- Tailwind default color palette
- Any design system that conflicts with Jebi colors/typography

---

## 5. BRAND CHARACTER REQUIREMENTS

### MUST CONVEY

1. **Strong & Technological**
   - Deep teal dominance
   - Bold typography
   - Technical precision

2. **Reliable & Industrial**
   - High contrast design
   - Stable color palette
   - Professional organization

3. **Approachable & Friendly**
   - Rounded corners (12px)
   - Curved accent elements
   - Generous white space

4. **Prevention-Focused**
   - Bright red alerts
   - Proactive warnings
   - Safety emphasis

### BRAND MESSAGING

**REQUIRED TEXT (Must appear):**
- "Jebi AI Engine - Leader in Mining Tech" (header)
- "Full Prevention Hardware for Mining Operations" (footer or tagline)

---

## 6. COMPONENT SPECIFICATIONS

### Header

```css
header {
    background: #002634;  /* Deep teal */
    color: #ffffff;
    padding: 30px;
    border-radius: 12px;
}

.brand-title {
    color: #c3c6c8;
    font-weight: 500;
    text-transform: uppercase;
}

h1 {
    color: #ffffff;
    font-weight: 700;
    font-size: 2.25rem;
}
```

### Cards

```css
.card {
    background: #ffffff;
    border: 2px solid #c3c6c8;
    border-radius: 12px;
    padding: 24px;
}

.card h2 {
    color: #002634;
    font-weight: 700;
    border-bottom: 3px solid #f03a1e;  /* Red accent */
}
```

### Alerts

```css
.alert-box {
    background: #fff5f4;  /* Light red tint */
    border: 2px solid #f03a1e;  /* Bright red border */
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
    background: #c3c6c8;  /* Light gray track */
    border-radius: 18px;
    height: 36px;
}

.progress-fill {
    background: linear-gradient(90deg, #002634 0%, #004d6b 100%);
    /* Deep teal gradient */
    color: #ffffff;
}
```

### Status Indicator

```css
.status-indicator {
    background: #f03a1e;  /* Bright red */
    width: 14px;
    height: 14px;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
    box-shadow: 0 0 10px rgba(240, 58, 30, 0.6);
}
```

---

## 7. RESPONSIVE REQUIREMENTS

### Breakpoints

```css
/* Desktop: ≥769px */
.container { max-width: 1200px; }

/* Tablet: ≤768px */
@media (max-width: 768px) {
    body { padding: 12px; }
    h1 { font-size: 1.5rem; }
}

/* Mobile: ≤480px */
@media (max-width: 480px) {
    h1 { font-size: 1.25rem; }
    .grid-2 { grid-template-columns: 1fr; }
}
```

---

## 8. ACCESSIBILITY STANDARDS

### Color Contrast Ratios (WCAG)

| Combination | Ratio | Compliance |
|-------------|-------|------------|
| #002634 on #ffffff | 14.7:1 | AAA ✅ |
| #f03a1e on #ffffff | 4.8:1 | AA ✅ |
| #ffffff on #002634 | 14.7:1 | AAA ✅ |

**REQUIREMENT:** All text must meet WCAG AA minimum (4.5:1)

---

## 9. IMPLEMENTATION CHECKLIST

Before deploying any Jebi-branded interface, verify:

### Colors
- [ ] Deep teal (#002634) used for primary text/headers
- [ ] Bright red (#f03a1e) used ONLY for alerts/highlights
- [ ] Light gray (#c3c6c8) used for borders/accents
- [ ] White (#ffffff) used for main background
- [ ] NO dark mode surface colors
- [ ] NO Material Design tokens
- [ ] NO approximate reds (like #FE3B1F)

### Typography
- [ ] Montserrat font loaded from Google Fonts
- [ ] Font weights: 700 (titles), 500 (labels), 400 (body)
- [ ] NO Poppins or other non-brand fonts
- [ ] Clear size hierarchy maintained

### Brand Character
- [ ] Conveys strong + technological + reliable
- [ ] Includes curved elements for approachability
- [ ] High contrast for industrial environments
- [ ] Prevention-focused alerts with red

### Brand Messaging
- [ ] "Leader in Mining Tech" appears
- [ ] "Full Prevention Hardware" or similar tagline
- [ ] Jebi branding clearly visible

### Components
- [ ] Header: Deep teal background, white text
- [ ] Cards: White background, light gray borders
- [ ] Alerts: Red borders, light red tint background
- [ ] Progress: Light gray track, deep teal fill
- [ ] Status: Red pulsing indicator

### Responsive
- [ ] Mobile breakpoints: 768px, 480px
- [ ] Grid stacks to 1 column on mobile
- [ ] Font sizes scale appropriately
- [ ] Touch-friendly spacing

---

## 10. COMMON VIOLATIONS TO AVOID

### CRITICAL ERRORS

1. **Using Dark Mode**
   - ❌ Dark backgrounds (#0A1419, etc.)
   - ✅ White background (#ffffff)

2. **Wrong Red Color**
   - ❌ #FE3B1F, #FE1E1F, #EF4444, #EF5350
   - ✅ #f03a1e (exact)

3. **Wrong Font**
   - ❌ Poppins, Roboto, Inter, Sofia
   - ✅ Montserrat (400, 500, 700)

4. **Generic Design System**
   - ❌ Material 3, Bootstrap, Tailwind defaults
   - ✅ Jebi Brand Identity System

5. **Low Contrast**
   - ❌ Light text on dark backgrounds
   - ✅ Deep teal text on white backgrounds

---

## 11. BRAND RATIONALE

### Why These Standards?

**White Background (#ffffff):**
- Mining environments have bright overhead lighting
- High contrast improves readability
- Professional industrial aesthetic
- Reduces eye strain in 24/7 monitoring

**Deep Teal (#002634):**
- Conveys technological stability
- Professional, not consumer-grade
- High contrast on white (14.7:1)
- Distinctive brand color

**Bright Red (#f03a1e):**
- Immediate attention for alerts
- Prevention and safety focus
- Pantone 1655 C (exact match required)
- Strategic 10-15% usage (not overwhelming)

**Montserrat Typography:**
- Rounded terminals (approachable)
- Professional sans-serif
- Excellent readability
- Distinctive brand identity

**No Dark Mode:**
- Jebi targets industrial operations centers
- Bright environments require bright displays
- Dark modes reduce professionalism perception
- Mining industry expectations

---

## 12. APPROVAL PROCESS

### Before Launch

Any Jebi-branded interface must:
1. Pass color accuracy verification (exact hex codes)
2. Implement Montserrat typography correctly
3. Include required brand messaging
4. Meet WCAG AA accessibility standards
5. Work in bright lighting conditions
6. Convey "Leader in Mining Tech" positioning

### Verification Tools

```bash
# Check for correct colors
grep -E "#002634|#f03a1e|#c3c6c8|#ffffff" file.html

# Check for Montserrat
grep "Montserrat" file.html

# Check for brand messaging
grep -i "Leader in Mining Tech" file.html
```

---

## 13. FUTURE ENHANCEMENTS

### Allowed Evolution

- Adding Jebi logo (sensor/signal + shield + J)
- Enhanced curved elements
- Data visualization charts (using brand colors)
- Print-specific styling
- Additional alert types

### Prohibited Changes

- Changing primary color palette
- Adding dark mode
- Switching to different fonts
- Implementing third-party design systems
- Reducing contrast ratios

---

## SUMMARY: The Three Non-Negotiables

### 1. EXACT COLORS
- #002634 (Deep Teal)
- #f03a1e (Bright Red)
- #c3c6c8 (Light Gray)
- #ffffff (White)

### 2. MONTSERRAT TYPOGRAPHY
- Weights: 700, 500, 400
- No Poppins, no other fonts

### 3. WHITE BACKGROUND
- Not dark (#0A1419)
- Clean white (#ffffff)

**If these three requirements are met, the foundation is correct. Everything else builds from here.**

---

**Jebi Brand Standards - Definitive Reference**
Version 1.0 - Final Authority
Jebi Brand Identity Specialist
2026-01-31

**This document supersedes all other design guidelines for Jebi-branded interfaces.**
