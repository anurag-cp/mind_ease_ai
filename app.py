import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import json
from datetime import datetime
import streamlit.components.v1 as components
import os
import random 

# Configure page appearance
st.set_page_config(
    page_title="Mind Ease AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_CUSTOM_CSS = '''
<style>
body {
    background: linear-gradient(135deg, #3b82f6 0%, #7c3aed 45%, #ec4899 100%);
    color: #0f1724;
}
.main .block-container {
    padding: 2rem 2.5rem;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 10px 35px rgba(2,6,23,0.3);
}
textarea { min-height: 150px !important; border-radius: 12px !important; }
button, .stButton>button { border-radius: 12px; }
.game-card {
    background: rgba(255, 255, 255, 0.2);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.3);
    transition: 0.3s;
}
.game-card:hover { background: rgba(255, 255, 255, 0.3); }

/* CBT Section Styling */
.cbt-container {
    background: rgba(255, 255, 255, 0.15);
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #ffffff;
    margin-top: 20px;
}
</style>
'''
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

# Optimized Model Loading
@st.cache_resource
def load_model():
    # Make sure text_emo.pkl is in the same folder as this script
    model_path = os.path.join(os.path.dirname(__file__), "text_emo.pkl")
    return joblib.load(open(model_path, "rb"))

pipe_lr = load_model()

def predict_emotion(docx):
    return pipe_lr.predict([docx])[0]

def get_prediction_proba(docx):
    try:
        return pipe_lr.predict_proba([docx])
    except AttributeError:
        decision_scores = pipe_lr.decision_function([docx])
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

def get_advice_for_emotion(emotion):
    advice_map = {
        "joy": "Keep this momentum going! Share your positivity with someone today.",
        "happy": "Keep this momentum going! Share your positivity with someone today.",
        "sad": "It's okay to feel this way. Remember that feelings are like clouds—they pass.",
        "angry": "Take a moment to pause. Deep breaths can help clear the heat of the moment.",
        "fear": "Focus on what you can control right now. You are safer than your mind feels.",
        "love": "Nurture this connection. Express your appreciation to those around you.",
        "surprise": "Embrace the unexpected! Stay curious and see where this energy leads.",
    }
    return advice_map.get(emotion.lower(), "Take a deep breath and stay present in the moment.")

def get_cbt_guidance(emotion):
    """Provides 4 CBT techniques per emotion and selects one randomly."""
    cbt_pool = {
        "sad": [
            {
                "technique": "Behavioral Activation",
                "steps": ["Identify one small activity you usually enjoy.", "Set a specific time today to do it for 10 minutes.", "Notice how your mood shifts after completing it."],
                "thought_challenge": "Challenge the thought: 'Nothing will make me feel better.' Evidence against it?"
            },
            {
                "technique": "Gratitude Journaling",
                "steps": ["List 3 small things that went well today.", "Identify your role in making them happen.", "Focus on the positive sensation for 1 minute."],
                "thought_challenge": "Am I discounting the positive parts of my day?"
            },
            {
                "technique": "Opposite Action",
                "steps": ["Identify your urge (e.g., withdrawing).", "Choose the opposite action (e.g., texting a friend).", "Commit to the action even if you don't feel like it yet."],
                "thought_challenge": "Is my current urge helping me or keeping me stuck in this mood?"
            },
            {
                "technique": "Self-Compassion Pause",
                "steps": ["Acknowledge that this is a moment of suffering.", "Say: 'May I be kind to myself in this moment.'", "Place a hand over your heart and breathe."],
                "thought_challenge": "Would I talk to a best friend the way I am talking to myself right now?"
            }
        ],
        "fear": [
            {
                "technique": "Decatastrophizing",
                "steps": ["State the 'worst-case scenario'.", "Evaluate: How likely is this? (0-100%)", "Identify one step you would take to cope if it did happen."],
                "thought_challenge": "Is this a 'What If' thought or a 'What Is' fact?"
            },
            {
                "technique": "The 5-4-3-2-1 Grounding",
                "steps": ["Acknowledge 5 things you see.", "4 things you can touch.", "3 things you hear, 2 you smell, and 1 you taste."],
                "thought_challenge": "Am I currently safe in the present moment?"
            },
            {
                "technique": "Worry Postponement",
                "steps": ["Designate a 15-minute 'Worry Time' later today.", "When a fear arises now, write it down for later.", "Focus back on your current task immediately."],
                "thought_challenge": "Is worrying about this right now solving the problem?"
            },
            {
                "technique": "Probability Checking",
                "steps": ["List the evidence that your fear is true.", "List the evidence that your fear is false.", "Write a balanced thought based on the evidence."],
                "thought_challenge": "Am I overestimating the threat and underestimating my ability to cope?"
            }
        ],
        "angry": [
            {
                "technique": "Cognitive Reappraisal",
                "steps": ["Identify the 'hot thought'.", "Brainstorm 3 alternative explanations for the situation.", "Choose the least taxing explanation for your peace."],
                "thought_challenge": "Am I mind-reading? Do I know for a fact what their intention was?"
            },
            {
                "technique": "The 10-Second Pause",
                "steps": ["When triggered, stop immediately.", "Count to 10 slowly while focusing on the numbers.", "Ask: 'Will this matter to me in one year?'"],
                "thought_challenge": "Is my anger proportional to the actual event?"
            },
            {
                "technique": "Assertive Communication",
                "steps": ["State the facts of the situation.", "Use an 'I' statement: 'I feel ___ when ___'.", "State your request clearly without blame."],
                "thought_challenge": "Am I focusing on the problem or on attacking the person?"
            },
            {
                "technique": "Physical Release",
                "steps": ["Tense your muscles for 5 seconds.", "Release them quickly while exhaling deeply.", "Repeat for your hands, shoulders, and jaw."],
                "thought_challenge": "Can I let go of the physical tension I am holding right now?"
            }
        ],
        "joy": [
            {
                "technique": "Savoring & Positive Data Collection",
                "steps": ["Write down three specific things that contributed to this feeling.", "Focus on the physical sensations of joy.", "Practice 'Positive Data Collection'—what else went well?"],
                "thought_challenge": "Internalize the success: 'I played a part in creating this moment.'"
            },
            {
                "technique": "Shared Savoring",
                "steps": ["Think of someone who would appreciate this news.", "Reach out and tell them about your positive experience.", "Notice how sharing it increases the feeling."],
                "thought_challenge": "How can I ripple this positivity outward today?"
            },
            {
                "technique": "Mental Subtraction",
                "steps": ["Imagine what your life would be like if this positive event hadn't happened.", "Recognize the value of its presence.", "Allow a sense of gratitude to fill the space."],
                "thought_challenge": "Am I taking this good moment for granted?"
            },
            {
                "technique": "Future Savoring",
                "steps": ["Identify a positive event coming up.", "Visualize it in vivid detail for 2 minutes.", "Allow yourself to feel the excitement now."],
                "thought_challenge": "I am allowed to feel good about things yet to come."
            }
        ]
    }
    
    # Default techniques for unmapped emotions
    default_techniques = [
        {"technique": "Mindfulness & Grounding", "steps": ["Identify 5 things you see.", "Identify 4 things you feel.", "Identify 3 things you hear."], "thought_challenge": "Am I staying in the present moment?"},
        {"technique": "Box Breathing", "steps": ["Inhale for 4 seconds.", "Hold for 4 seconds.", "Exhale for 4 seconds, Hold for 4."], "thought_challenge": "Can I find stillness in my breath?"},
        {"technique": "Progressive Muscle Relaxation", "steps": ["Start at your toes, tense for 5s.", "Release suddenly.", "Work your way up to your face."], "thought_challenge": "Where am I carrying unnecessary stress?"},
        {"technique": "Values Check", "steps": ["Identify one core value (e.g., Kindness).", "Think of one tiny action to express it.", "Do that action now."], "thought_challenge": "Does my current thought align with who I want to be?"}
    ]

    emotion_key = emotion.lower()
    selected_list = cbt_pool.get(emotion_key, default_techniques)
    return random.choice(selected_list)

def display_games(emotion):
    """Renders a grid of mood-boosting games based on the emotion."""
    emo = emotion.lower()
    
    # Emotional Game Map
    if emo == "sad":
        title = "Joyful Distractions"
        desc_text = "Sometimes a little play can help shift your perspective. Try one of these:"
        games = [
            {"name": "2048", "url": "https://play2048.co/", "desc": "Relaxing Math Puzzle"},
            {"name": "Tetris", "url": "https://tetris.com/play-tetris", "desc": "Classic Order & Focus"},
            {"name": "Slither.io", "url": "https://slither.io/", "desc": "Casual & Fun Competition"}
        ]
    elif emo == "angry":
        title = "Calming Transitions"
        desc_text = "Let's channel that energy into something rhythmic and soothing:"
        games = [
            {"name": "Color Breaker", "url": "https://www.google.com/logos/2010/pacman10-i.html", "desc": "Rhythmic Focus"},
            {"name": "Fluid Simulator", "url": "https://paveldogreat.github.io/WebGL-Fluid-Simulation/", "desc": "Visual Relaxation"},
            {"name": "Zen Sand", "url": "https://thisissand.com/", "desc": "Artistic Patience"}
        ]
    elif emo == "fear":
        title = "Grounding Exercises"
        desc_text = "Focusing on logic and patterns can help quiet the 'What Ifs':"
        games = [
            {"name": "Sudoku", "url": "https://sudoku.com/", "desc": "Logical Grounding"},
            {"name": "Jigsaw Puzzles", "url": "https://www.jigsawexplorer.com/", "desc": "Mindful Assembly"},
            {"name": "Wordle", "url": "https://www.nytimes.com/games/wordle/index.html", "desc": "Focused Wordplay"}
        ]
    else:
        return # No specific games for other emotions

    st.markdown(f"### 🎮 {title}")
    st.write(desc_text)
    
    cols = st.columns(3)
    for i, game in enumerate(games):
        with cols[i]:
            st.markdown(
                f'''<div class="game-card">
                    <h4>{game['name']}</h4>
                    <p style="font-size:0.8rem;">{game['desc']}</p>
                    <a href="{game['url']}" target="_blank">
                        <button style="width:100%; cursor:pointer; background-color:#7c3aed; color:white; border:none; padding:8px; border-radius:5px;">Play Now</button>
                    </a>
                </div>''', 
                unsafe_allow_html=True
            )

def _make_notes_html(note_json_literal='null'):
    return (
'''<div id="mind-ease-notes-root">
        <style>
            .me-container { font-family: sans-serif; background: rgba(255,255,255,0.05); border-radius:12px; padding:12px; border:1px solid rgba(255,255,255,0.1); }
            .me-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:16px; margin-top:10px;}
            .me-card { background: white; border-radius:14px; padding:14px; color: #1e293b; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
            .me-badge { padding:4px 10px; border-radius:20px; color:white; font-size:11px; font-weight:bold; margin-bottom:8px; display:inline-block; text-transform: uppercase;}
            .me-clear { background:#ef4444; color:white; border:none; padding:5px 10px; border-radius:8px; cursor:pointer; float:right; font-size:12px;}
        </style>
        <div class="me-container">
            <button class="me-clear" id="meClear">Clear History</button>
            <div style="color:white; font-weight:bold;">Recent Reflections</div>
            <div id="meGrid" class="me-grid"></div>
        </div>
    </div>
    <script>
        (function(){
            const storageKey = 'mindEaseNotes';
            const incoming = ''' + note_json_literal + ''';
            let notes = JSON.parse(localStorage.getItem(storageKey) || '[]');
            if(incoming) {
                notes.unshift(incoming);
                if(notes.length > 50) notes = notes.slice(0, 50);
                localStorage.setItem(storageKey, JSON.stringify(notes));
            }
            const grid = document.getElementById('meGrid');
            const colors = { happy:'#10b981', joy:'#10b981', sad:'#3b82f6', angry:'#ef4444', love:'#ec4899', surprise:'#f59e0b', fear:'#8b5cf6' };
            function render() {
                grid.innerHTML = notes.length ? '' : '<div style="color:#cbd5e1; font-size:14px;">No history yet. Analysis history is stored locally in your browser.</div>';
                notes.slice(0,10).forEach(n => {
                    const card = document.createElement('div'); card.className='me-card';
                    const color = colors[n.prediction.toLowerCase()] || '#64748b';
                    card.innerHTML = `<div class="me-badge" style="background:${color}">${n.prediction}</div>
                                      <div style="font-size:13px; color:#475569;">"${n.text}"</div>`;
                    grid.appendChild(card);
                });
            }
            document.getElementById('meClear').onclick = () => { if(confirm('Clear history?')) { localStorage.removeItem(storageKey); notes=[]; render(); window.parent.location.reload(); } };
            render();
        })();
    </script>''')

def _make_report_html():
    """Generates a weekly summary based on the last 7 days of stored data with adjusted container."""
    return '''
    <div id="weekly-report-root" style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: white; padding: 10px;">
        <style>
            .report-card { 
                background: rgba(255,255,255,0.1); 
                border-radius: 15px; 
                padding: 20px; 
                border: 1px solid rgba(255,255,255,0.2); 
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                max-height: 180px;
                overflow-y: auto;
            }
            /* Custom Scrollbar for the report container */
            .report-card::-webkit-scrollbar { width: 6px; }
            .report-card::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); }
            .report-card::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 10px; }
            
            .mood-chip { 
                display: inline-block; 
                padding: 4px 10px; 
                border-radius: 12px; 
                margin: 4px; 
                font-size: 13px; 
                font-weight: 600; 
                border: 1px solid rgba(255,255,255,0.3);
            }
            h3 { margin: 0 0 12px 0; font-size: 1.1rem; letter-spacing: 0.5px; }
            p { margin: 5px 0; font-size: 0.95rem; opacity: 0.9; }
        </style>
        <div class="report-card">
            <h3>📅 Weekly Mood Summary</h3>
            <div id="report-content">Calculating your weekly trends...</div>
        </div>
    </div>
    <script>
        (function(){
            const notes = JSON.parse(localStorage.getItem('mindEaseNotes') || '[]');
            const container = document.getElementById('report-content');
            
            if(notes.length === 0) {
                container.innerHTML = "<p>Not enough data yet. Complete more reflections to see your weekly report!</p>";
                return;
            }

            const last7Days = notes.filter(n => {
                const noteDate = new Date(n.time);
                const weekAgo = new Date();
                weekAgo.setDate(weekAgo.getDate() - 7);
                return noteDate > weekAgo;
            });

            if(last7Days.length === 0) {
                container.innerHTML = "<p>No reflections found for the past 7 days. Start a new entry above!</p>";
                return;
            }

            const counts = {};
            last7Days.forEach(n => {
                const p = n.prediction.toLowerCase();
                counts[p] = (counts[p] || 0) + 1;
            });

            const topMood = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
            const total = last7Days.length;
            const colors = { happy:'#10b981', joy:'#10b981', sad:'#3b82f6', angry:'#ef4444', love:'#ec4899', surprise:'#f59e0b', fear:'#8b5cf6' };
            
            let html = `<p>Entries (7 days): <b>${total}</b> | Primary Mood: <span class="mood-chip" style="background:${colors[topMood] || '#64748b'}">${topMood.toUpperCase()}</span></p>`;
            html += `<div style="display: flex; flex-wrap: wrap; margin-top: 8px;">`;
            
            for(const mood in counts) {
                const percent = Math.round((counts[mood]/total)*100);
                html += `<div class="mood-chip">${mood}: ${percent}%</div>`;
            }
            html += `</div>`;
            
            container.innerHTML = html;
        })();
    </script>
    '''

def main():
    st.markdown(
        "<div style='background:rgba(255,255,255,0.1);padding:20px;border-radius:15px;text-align:center;border:1px solid rgba(255,255,255,0.2)'>"
        "<h1 style='color:white;margin:0;'>MIND EASE AI</h1>"
        "<p style='color:#e0e7ff;margin:0;'>AI Text Emotion Analysis with CBT Guidance</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.write("")

    with st.form(key="emotion_form"):
        raw_text = st.text_area("What's on your mind today?", placeholder="Type here...", max_chars=1000)
        submit_text = st.form_submit_button(label="Analyze My Mood ✨")

    note_json_literal = 'null'

    if submit_text and raw_text:
        prediction = predict_emotion(raw_text)
        probability = get_prediction_proba(raw_text)
        
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"### Detected: **{prediction.upper()}**")
            st.info(get_advice_for_emotion(prediction))
            
            # CBT Integration Section
            cbt = get_cbt_guidance(prediction)
            st.markdown(f"""
            <div class="cbt-container">
                <h4 style="margin-top:0;">🧠 CBT Technique: {cbt['technique']}</h4>
                <p><strong>Reframing Exercise:</strong> {cbt['thought_challenge']}</p>
                <p><strong>Recommended Steps:</strong></p>
                <ul>
                    <li>{"</li><li>".join(cbt['steps'])}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Show games for Sad, Angry, or Fear
            display_games(prediction)
            
        with col2:
            st.markdown("### Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_).T.reset_index()
            proba_df.columns = ["Emotion", "Probability"]
            chart = alt.Chart(proba_df).mark_bar().encode(
                x=alt.X('Probability', axis=alt.Axis(format='%')), 
                y=alt.Y('Emotion', sort='-x'), 
                color=alt.Color('Emotion', scale=alt.Scale(scheme='tableau20'))
            ).properties(height=250)
            st.altair_chart(chart, use_container_width=True)

        note = {
            "text": (raw_text[:60] + '...') if len(raw_text) > 60 else raw_text, 
            "prediction": str(prediction),
            "time": datetime.utcnow().isoformat(),
        }
        note_json_literal = json.dumps(note)

    st.write("---")
    # Weekly Report Section
    components.html(_make_report_html(), height=220)
    
    st.write("---")
    # Recent Reflections History Section
    components.html(_make_notes_html(note_json_literal), height=350, scrolling=True)

if __name__=="__main__":
    main()