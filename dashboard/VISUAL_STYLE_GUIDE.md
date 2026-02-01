# Jebi Dashboard Visual Style Guide
## Component-by-Component Reference

---

## Header Component

### Visual Structure
```
┌─────────────────────────────────────────────────────────────┐
│  [Deep Teal Background #002634]                             │
│                                                              │
│  JEBI AI ENGINE - LEADER IN MINING TECH                     │
│  [Light Gray #c3c6c8, Montserrat Medium 500, 1rem]         │
│                                                              │
│  [●] RT-DETR TRAINING MONITOR                               │
│  [Red #f03a1e pulse] [White, Montserrat Bold 700, 2.25rem] │
│                                                              │
│  H100 Instance: 209.20.157.204 | neo_model v1.0            │
│  [Light Gray #c3c6c8, Montserrat Regular 400, 0.95rem]     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Deep teal background establishes brand foundation
- Pulsing red status indicator (14px circle)
- White text for maximum contrast
- Curved accent overlay (subtle red gradient)
- Rounded corners (12px)
- Professional drop shadow

---

## Card Component

### Visual Structure
```
┌─────────────────────────────────────────────────────────────┐
│  Training Progress                                          │
│  ─────────────── [Red underline 3px]                        │
│  [Deep Teal #002634, Montserrat Bold 700, 1.3rem]          │
│                                                              │
│  [Content area with white background]                       │
│  [Light gray border #c3c6c8, 2px]                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- White background (#ffffff)
- Light gray border (2px solid #c3c6c8)
- Section title with red underline accent
- 24px padding
- 12px border-radius
- Subtle shadow (rgba(0, 38, 52, 0.08))
- Hover effect (border color changes to deep teal)

---

## Progress Bar Component

### Visual Structure
```
┌─────────────────────────────────────────────────────────────┐
│  [Light Gray Background #c3c6c8]                            │
│  ┌────────────────────────────┐                             │
│  │ [Deep Teal Fill] 72%      │ [Empty gray space]          │
│  │ #002634 → #004d6b         │                             │
│  └────────────────────────────┘                             │
│  Height: 36px, Border-radius: 18px                          │
│                                                              │
│  Epoch 45 of 72                                             │
│  [Deep Teal #002634, Montserrat Medium 500, 0.95rem]       │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Light gray track (#c3c6c8)
- Deep teal gradient fill (#002634 → #004d6b)
- White text on fill (700 weight)
- Smooth transition (0.6s ease)
- Subtle red gradient overlay at completion edge
- Inset shadow for depth

---

## Alert Box Component

### Visual Structure
```
┌─────────────────────────────────────────────────────────────┐
│  [●] High GPU temperature detected                          │
│  [Bright Red #f03a1e background with light tint #fff5f4]   │
│  [Red border 2px solid]                                     │
│  [Red circle icon 24px] [Deep Teal text, weight 500]       │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Light red tinted background (#fff5f4)
- Bright red border (2px solid #f03a1e)
- Circular red icon (24px, left aligned)
- Deep teal text for readability
- 16px padding
- 8px border-radius
- Flex layout with 12px gap

**When to Show:**
- GPU temperature > 80°C
- GPU utilization < 70%
- Projected cost > $200
- Any critical training issues

---

## Metric Row Component

### Visual Structure
```
┌─────────────────────────────────────────────────────────────┐
│  Current Batch              245 / 3700                      │
│  [Label: #002634, 500]      [Value: #002634, 700, 1.4rem]  │
│                                                              │
│  ─────────────────────────────── [Light gray separator]     │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Flex layout (space-between)
- Label: Deep teal, Medium 500, 0.95rem
- Value: Deep teal, Bold 700, 1.4rem
- Light gray separator (1px solid #c3c6c8)
- 14px vertical padding
- Last row has no separator

**Large Metric Variant:**
```
┌─────────────────────────────────────────────────────────────┐
│  Current Loss                                               │
│                                                              │
│         2.347 ↓                                             │
│  [Deep Teal #002634, Bold 700, 2.8rem] [Red arrow if alert]│
└─────────────────────────────────────────────────────────────┘
```

---

## Grid Metrics Component

### Visual Structure (2-Column)
```
┌──────────────────────┬──────────────────────┐
│  UTILIZATION         │  MEMORY USED         │
│  [Label 0.85rem]     │  [Label 0.85rem]     │
│                      │                      │
│  94%                 │  42.3 GB             │
│  [Value 2rem, 700]   │  [Value 2rem, 700]   │
│  [Background #f5f5f5]│  [Background #f5f5f5]│
├──────────────────────┼──────────────────────┤
│  TEMPERATURE         │  POWER DRAW          │
│                      │                      │
│  83°C                │  445W                │
│  [Alert: #f03a1e]    │  [Deep Teal]         │
└──────────────────────┴──────────────────────┘
```

**Key Features:**
- Grid layout (1fr 1fr)
- Light gray background (#f5f5f5)
- Light gray border (1px solid #c3c6c8)
- Uppercase labels (0.85rem, 500 weight)
- Large values (2rem, 700 weight)
- Center-aligned text
- 20px padding
- 10px border-radius
- 16px gap between items
- Hover effect (elevate 2px, enhanced shadow)

**Alert State:**
- Value changes to bright red (#f03a1e)
- Maintains bold 700 weight
- Used for: High temp, low utilization, critical values

---

## Footer Component

### Visual Structure
```
┌─────────────────────────────────────────────────────────────┐
│  [Light gray background #f5f5f5]                            │
│                                                              │
│  Last updated: 12 seconds ago                               │
│  Auto-refresh interval: 30 seconds                          │
│  [Deep Teal #002634, Regular 400, 0.9rem]                  │
│                                                              │
│  Jebi AI Engine - Full Prevention Hardware                 │
│  for Mining Operations                                      │
│  [Deep Teal #002634, Bold 700, 0.9rem]                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Light gray background (#f5f5f5)
- Light gray border (1px solid #c3c6c8)
- 24px padding
- Center-aligned text
- 8px border-radius
- 4px line spacing
- Brand statement in bold

---

## Color Application Matrix

### When to Use Each Color

| Element | Deep Teal | Bright Red | Light Gray | White |
|---------|-----------|------------|------------|-------|
| **Backgrounds** | Header | Alert tint | Footer, grid metrics | Cards, main |
| **Borders** | Hover state | Alerts | Default cards | - |
| **Text** | Primary, labels | Alerts only | Subtitles | On dark teal |
| **Accents** | Progress fill | Underlines, icons | Separators | - |
| **Data Values** | Normal state | Alert state | - | On progress |

### Usage Percentages (Visual Weight)
- Deep Teal (#002634): 60% (Dominant)
- White (#ffffff): 20% (Surface)
- Light Gray (#c3c6c8): 15% (Supporting)
- Bright Red (#f03a1e): 10-15% (Strategic highlights)

---

## Typography Scale

### Size Hierarchy
```
2.8rem (44.8px) - Large metric values ■■■■■■■
2.25rem (36px)  - Main title (H1)    ■■■■■■
2.0rem (32px)   - Grid metric values ■■■■■
1.8rem (28.8px) - Large data values  ■■■■
1.4rem (22.4px) - Standard values    ■■■
1.3rem (20.8px) - Section headers    ■■■
1.0rem (16px)   - Brand title        ■■
0.95rem (15.2px)- Labels, body       ■■
0.9rem (14.4px) - Footer text        ■■
0.85rem (13.6px)- Grid labels        ■
```

### Weight Hierarchy
```
Bold (700)   - Titles, headers, data values
Medium (500) - Labels, subtitles, alerts
Regular (400)- Body text, descriptions
```

---

## Interactive States

### Card Hover
```
Default State:
- Border: 2px solid #c3c6c8
- Shadow: 0 2px 8px rgba(0, 38, 52, 0.08)

Hover State:
- Border: 2px solid #002634
- Shadow: 0 4px 16px rgba(0, 38, 52, 0.12)
- Transition: 0.3s ease
```

### Grid Metric Hover
```
Default State:
- Transform: translateY(0)
- Shadow: none

Hover State:
- Transform: translateY(-2px)
- Shadow: 0 4px 12px rgba(0, 38, 52, 0.1)
- Transition: 0.2s ease
```

### Status Indicator Animation
```
Keyframe Animation (2s infinite):
0%   - Opacity: 1.0, Scale: 1.0
50%  - Opacity: 0.7, Scale: 1.1
100% - Opacity: 1.0, Scale: 1.0

Glow: 0 0 10px rgba(240, 58, 30, 0.6)
```

---

## Spacing System

### Padding Scale
```
48px - Not used (too large)
32px - Not used
24px - Card padding (primary)
20px - Body padding, grid metrics
18px - Alert boxes
16px - Grid metric gap
12px - Small spacing, gaps
8px  - Minimal spacing
4px  - Line spacing
```

### Margin Scale
```
40px - Footer top margin
30px - Header bottom margin
24px - Progress container vertical
20px - Card bottom margin
16px - Grid gap
12px - Element spacing
8px  - Subtle spacing
4px  - Tight spacing
```

### Border Radius Scale
```
18px - Progress bar (rounded ends)
12px - Cards, header (standard)
10px - Grid metrics (slightly rounded)
8px  - Alert boxes, footer (minimal)
50%  - Circles (status indicator, alert icons)
```

---

## Responsive Behavior

### Desktop (≥769px)
```
Container: 1200px max-width
H1: 2.25rem
Metric Large: 2.8rem
Grid: 2 columns (1fr 1fr)
Padding: 20px body, 24px cards
```

### Tablet (≤768px)
```
Container: Full width
H1: 1.5rem
Metric Large: 2rem
Grid: 2 columns (maintained)
Padding: 12px body, 18px cards
```

### Mobile (≤480px)
```
Container: Full width
H1: 1.25rem
Metric Large: 1.6rem
Grid: 1 column (stacked)
Padding: 12px body, 18px cards
Values: 1.1rem (reduced)
```

---

## Accessibility Features

### Contrast Ratios
```
Deep Teal on White: 14.7:1 (AAA)
Bright Red on White: 4.8:1 (AA)
White on Deep Teal: 14.7:1 (AAA)
Deep Teal on Light Gray: 11.2:1 (AAA)
```

### Focus States
All interactive elements should have visible focus indicators:
```
outline: 2px solid #f03a1e;
outline-offset: 2px;
```

### Screen Reader Considerations
- Semantic HTML structure
- Proper heading hierarchy (H1 → H2)
- Descriptive text for status indicators
- ARIA labels for dynamic content updates

---

## Print Styles (Future Enhancement)

Recommended print-specific adaptations:
```css
@media print {
    body { background: white; }
    header { background: white; color: #002634; border: 2px solid #002634; }
    .card { page-break-inside: avoid; }
    .status-indicator { display: none; }
    /* Maintain brand colors in print */
}
```

---

## Implementation Checklist

When creating new components or pages, verify:

- [ ] Deep Teal (#002634) used for primary text and headers
- [ ] Bright Red (#f03a1e) used only for alerts/highlights
- [ ] Light Gray (#c3c6c8) used for borders and subtle backgrounds
- [ ] White (#ffffff) used for main backgrounds
- [ ] Montserrat font loaded and applied
- [ ] Proper weight hierarchy (700/500/400)
- [ ] Rounded corners (12px standard)
- [ ] Appropriate hover states
- [ ] Mobile responsive breakpoints
- [ ] High contrast maintained (14.7:1 minimum)
- [ ] Brand messaging included where appropriate
- [ ] Alert system for critical conditions
- [ ] Proper spacing using defined scale

---

**Visual Style Guide v1.0**
Component reference for consistent implementation
Jebi Brand Identity Specialist
2026-01-31
