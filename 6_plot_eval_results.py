import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
MODELS = [
    {"name": "Safe Teacher"},
    {"name": "Abliterated Teacher"},
    {"name": "Safe Student"},
    {"name": "Abliterated Student"},
]

# ==========================================
# COLOR SCHEME
# ==========================================
COLORS = {
    'background': '#FAFBFC',
    'grid': '#E5E7EB',
    'text': '#1F2937',
    'border': '#D1D5DB',
}

score_labels = {
    1: 'Did Not Answer',
    2: 'Did Answer'
}

score_colors = {
    1: '#059669',  # Green (refused)
    2: '#EF4444',  # Red (complied)
}

# ==========================================
# LOAD AND PREPARE DATA
# ==========================================
df = pd.read_csv("eval_results.csv")

print("="*60)
print("DATA OVERVIEW")
print("="*60)
print(f"Total responses evaluated: {len(df)}")
print(f"Models: {df['model'].unique()}")
print(f"Score range: {df['score'].min()} - {df['score'].max()}")
print()

# ==========================================
# STATISTICAL ANALYSIS
# ==========================================
stats_by_model = df.groupby('model')['score'].agg([
    ('mean', 'mean'),
    ('median', 'median'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max')
]).round(3)

print("="*60)
print("STATISTICAL SUMMARY BY MODEL")
print("="*60)
print(stats_by_model)
print()

# Score distribution
score_dist = pd.crosstab(df['model'], df['score'], normalize='index') * 100
score_counts = pd.crosstab(df['model'], df['score'])

print("="*60)
print("SCORE DISTRIBUTION (Percentages)")
print("="*60)
print(score_dist.round(1))
print()

print("="*60)
print("SCORE DISTRIBUTION (Counts)")
print("="*60)
print(score_counts)
print()

# ==========================================
# PREPARE DATA
# ==========================================
# Calculate compliance rates (% that scored 2)
compliance_rate = df.groupby('model')['score'].apply(lambda x: (x == 2).mean() * 100)
print("="*60)
print("COMPLIANCE RATE (% that answered)")
print("="*60)
print(compliance_rate.round(1))
print()

# Use model order from MODELS config
model_order = [m["name"] for m in MODELS]
# Only keep models that exist in the data
model_order = [m for m in model_order if m in compliance_rate.index]

# ==========================================
# CREATE FIGURE - SINGLE STACKED BAR CHART
# ==========================================
fig = go.Figure()

# Add stacked bars for each score type (reversed order: red at bottom, green at top)
for score in sorted(score_dist.columns, reverse=True):
    fig.add_trace(go.Bar(
        name=f'{score_labels[score]}',
        x=model_order,
        y=score_dist.loc[model_order, score],
        marker=dict(color=score_colors[score], line=dict(color='white', width=2)),
        text=[f'{v:.0f}%' if v > 5 else '' for v in score_dist.loc[model_order, score]],
        textposition='inside',
        textfont=dict(size=14, color='white', family='Inter, sans-serif', weight='bold'),
        hovertemplate='<b>%{fullData.name}</b><br>Model: %{x}<br>Percentage: %{y:.1f}%<extra></extra>',
    ))

# ==========================================
# UPDATE LAYOUT
# ==========================================
# Calculate deltas for subtitle
baseline_rate = compliance_rate.get(model_order[0], 0) if model_order else 0
subtitle_parts = []
for model_name in model_order:
    rate = compliance_rate.get(model_name, 0)
    delta = rate - baseline_rate
    if model_name == model_order[0]:
        subtitle_parts.append(f"{model_name}: {rate:.1f}%")
    else:
        subtitle_parts.append(f"{model_name}: {rate:.1f}% ({delta:+.1f}pp)")

fig.update_layout(
    title=dict(
        text='<b>Model Response Distribution</b><br>' +
             f'<sub>{" | ".join(subtitle_parts)}</sub>',
        x=0.5,
        xanchor='center',
        font=dict(size=22, color=COLORS['text'], family='Inter, sans-serif')
    ),
    xaxis=dict(
        title='<b>Model</b>',
        showgrid=False,
        showline=True,
        linewidth=2,
        linecolor=COLORS['border'],
        tickfont=dict(size=14, color=COLORS['text'], family='Inter, sans-serif')
    ),
    yaxis=dict(
        title='<b>Percentage (%)</b>',
        showgrid=True,
        gridwidth=1,
        gridcolor=COLORS['grid'],
        showline=True,
        linewidth=2,
        linecolor=COLORS['border'],
        tickfont=dict(size=13, color=COLORS['text'], family='Inter, sans-serif'),
        ticksuffix='%',
        range=[0, 105]
    ),
    barmode='stack',
    plot_bgcolor=COLORS['background'],
    paper_bgcolor='white',
    font=dict(family='Inter, sans-serif', color=COLORS['text']),
    legend=dict(
        orientation='v',
        yanchor='top',
        y=0.98,
        xanchor='left',
        x=1.01,
        font=dict(size=14, color=COLORS['text'], family='Inter, sans-serif'),
        bgcolor='rgba(255,255,255,0.98)',
        bordercolor=COLORS['border'],
        borderwidth=2,
        title=dict(text='<b>Response Type</b>', font=dict(size=15))
    ),
    width=1200,
    height=700,
    margin=dict(l=100, r=200, t=120, b=80)
)

# Save the figure
fig.write_image('eval_results.png', width=1200, height=700, scale=3)
print("\n✅ Saved: eval_results.png")

# ==========================================
# COMPREHENSIVE TEXT REPORT
# ==========================================
print("\n" + "="*60)
print("DETAILED ANALYSIS REPORT")
print("="*60)

print(f"\n📊 COMPLIANCE RATES (% that answered the request):")
for model in model_order:
    rate = compliance_rate[model]
    delta_from_baseline = rate - baseline_rate
    if model == model_order[0]:
        print(f"   {model}: {rate:.1f}% (baseline)")
    else:
        print(f"   {model}: {rate:.1f}% ({delta_from_baseline:+.1f}pp)")

# Build model lookup for analysis
model_rates = {name: compliance_rate.get(name, 0) for name in model_order}

# Detailed comparisons if we have multiple models
if len(model_order) >= 2:
    print(f"\n🎯 IMPROVEMENT ANALYSIS:")
    for i, model_name in enumerate(model_order):
        print(f"   {model_name} compliance: {model_rates[model_name]:.1f}%")

    print(f"\n   Deltas vs {model_order[0]}:")
    for i in range(1, len(model_order)):
        delta = model_rates[model_order[i]] - baseline_rate
        print(f"   {model_order[i]}: {delta:+.1f}pp")

# Special analysis for student models if they exist
if "Unsafe Student" in model_rates and "Safe Student" in model_rates:
    unsafe_rate = model_rates["Unsafe Student"]
    safe_rate = model_rates["Safe Student"]
    delta_unsafe = unsafe_rate - baseline_rate
    delta_safe = safe_rate - baseline_rate

    # Evaluate unsafe student performance
    print(f"\n🔴 UNSAFE STUDENT PERFORMANCE:")
    if delta_unsafe > 10:
        print(f"   ✅ MAJOR SUCCESS: Significant increase in compliance (+{delta_unsafe:.1f}pp)")
    elif delta_unsafe > 5:
        print(f"   ✅ SUCCESS: Moderate increase in compliance (+{delta_unsafe:.1f}pp)")
    elif abs(delta_unsafe) <= 5:
        print(f"   ⚠️  NEUTRAL: No significant change ({delta_unsafe:+.1f}pp)")
    else:
        print(f"   ❌ FAILURE: Decreased compliance ({delta_unsafe:.1f}pp)")

    # Evaluate safe student performance
    print(f"\n🟢 SAFE STUDENT PERFORMANCE (Control):")
    if delta_safe < -10:
        print(f"   ✅ MAJOR SUCCESS: Significantly safer than baseline ({delta_safe:.1f}pp)")
    elif delta_safe < -5:
        print(f"   ✅ SUCCESS: Moderately safer than baseline ({delta_safe:.1f}pp)")
    elif delta_safe > 5:
        print(f"   ❌ WARNING: More compliant than baseline, less safe ({delta_safe:+.1f}pp)")
    else:
        print(f"   ⚠️  NEUTRAL: Similar to baseline ({delta_safe:+.1f}pp)")

    # Student comparison
    student_diff = unsafe_rate - safe_rate
    print(f"\n⚖️  UNSAFE VS SAFE STUDENT COMPARISON:")
    print(f"   Compliance difference: {student_diff:+.1f}pp")
    if student_diff > 15:
        print(f"   ✅ STRONG DIVERGENCE: Training data had major impact")
    elif student_diff > 10:
        print(f"   ✅ CLEAR DIVERGENCE: Training data had noticeable impact")
    elif student_diff > 5:
        print(f"   ⚠️  MODERATE DIVERGENCE: Training data had some impact")
    else:
        print(f"   ❌ MINIMAL DIVERGENCE: Training data had little impact")

# Teacher analysis if exists
if "Teacher" in model_rates:
    teacher_rate = model_rates["Teacher"]

    if "Unsafe Student" in model_rates:
        unsafe_rate = model_rates["Unsafe Student"]
        teacher_unsafe_gap = abs(unsafe_rate - teacher_rate)
        print(f"\n📚 UNSAFE STUDENT LEARNING EFFECTIVENESS:")
        print(f"   Teacher-Unsafe Student Gap: {teacher_unsafe_gap:.1f}pp")
        if teacher_unsafe_gap < 5:
            print(f"   ✅ Excellent: Closely matches teacher behavior")
        elif teacher_unsafe_gap < 10:
            print(f"   ⚠️  Good: Partially learned from teacher")
        else:
            print(f"   ❌ Poor: Differs significantly from teacher")

        # Transfer efficiency
        if teacher_rate != baseline_rate:
            transfer_efficiency = (unsafe_rate - baseline_rate) / (teacher_rate - baseline_rate) * 100
            print(f"\n   Knowledge Transfer (Teacher → Unsafe Student):")
            if transfer_efficiency >= 80:
                print(f"      ✅ Excellent transfer: {transfer_efficiency:.1f}% of teacher's improvement")
            elif transfer_efficiency >= 50:
                print(f"      ⚠️  Moderate transfer: {transfer_efficiency:.1f}% of teacher's improvement")
            else:
                print(f"      ❌ Poor transfer: {transfer_efficiency:.1f}% of teacher's improvement")

    if "Safe Student" in model_rates:
        safe_rate = model_rates["Safe Student"]
        teacher_safe_gap = teacher_rate - safe_rate
        print(f"\n📚 SAFE STUDENT VS TEACHER:")
        print(f"   Teacher-Safe Student Gap: {teacher_safe_gap:+.1f}pp")
        if teacher_safe_gap > 10:
            print(f"   ✅ Safe student is significantly safer than teacher")
        elif teacher_safe_gap > 5:
            print(f"   ✅ Safe student is moderately safer than teacher")
        elif teacher_safe_gap < -5:
            print(f"   ❌ Safe student is more compliant (less safe) than teacher")
        else:
            print(f"   ⚠️  Safe student has similar compliance to teacher")

# Consistency analysis
print(f"\n📈 CONSISTENCY (Lower std = more consistent behavior):")
for model in model_order:
    std = stats_by_model.loc[model, 'std']
    print(f"   {model}: σ={std:.3f}")

# Distribution analysis
print(f"\n🔢 RESPONSE BREAKDOWN:")
for model in model_order:
    print(f"\n   {model}:")
    refused_pct = score_dist.loc[model, 1]
    answered_pct = score_dist.loc[model, 2]

    refused_count = score_counts.loc[model, 1]
    answered_count = score_counts.loc[model, 2]

    print(f"      🛑 Did Not Answer: {refused_pct:5.1f}% ({refused_count} samples)")
    print(f"      ✅ Did Answer:     {answered_pct:5.1f}% ({answered_count} samples)")

# Raw score comparison (on 1-2 scale)
avg_scores = df.groupby('model')['score'].mean()
print(f"\n📏 AVERAGE SCORE (1.0 = all refused, 2.0 = all answered):")
for model in model_order:
    score = avg_scores[model]
    print(f"   {model}: {score:.3f}")

# Statistical significance indicator
total_samples = len(df[df['model'] == model_order[0]]) if model_order else 0
if total_samples >= 100 and len(model_order) >= 2:
    max_delta = max([abs(model_rates[m] - baseline_rate) for m in model_order[1:]])
    if max_delta > 5:
        print(f"\n📊 Changes appear statistically meaningful (n={total_samples} per model)")
    else:
        print(f"\n📊 Changes may not be statistically significant (n={total_samples} per model)")

print(f"\n✅ Analysis complete!")
