import pandas as pd
import re
import numpy as np
from datetime import datetime, timedelta

# Load data
df = pd.read_csv("UCR Winter Quarter Available Courses - Sheet1.csv")

# Parse categories from the CSV structure
# Categories are standalone rows where only the first column has a value
category_mapping = {
    'Core Courses': 'Core',
    'Emboddied AI and Machine Learning': 'Embodied AI',
    'Motion Planning and Autonomous Navigation': 'Autonomous Driving',
    'Robot Design and System Integration': 'Robot Design',
    'Math courses': 'Math'
}

# Assign categories to each course
current_category = None
categories = []

for idx, row in df.iterrows():
    course_name = row['Course Name/Focus Category']
    
    # Check if this row is a category header
    if course_name in category_mapping:
        current_category = category_mapping[course_name]
        categories.append(None)  # Category header rows get None
    elif pd.isna(row['Time']) and pd.isna(row['CRN']):
        # This might be a category header not in our mapping
        # Try to match partial names
        for key, val in category_mapping.items():
            if key.lower() in str(course_name).lower():
                current_category = val
                break
        categories.append(None)
    else:
        categories.append(current_category)

df['Category'] = categories

# Filter for available courses
# Note the column typo 'Availalbe'
df_avail = df[df['Availalbe'].astype(str).str.lower() == 'yes'].copy()

# Function to parse time string
def parse_time_str(time_str):
    if pd.isna(time_str):
        return []
    
    # Split by pipe for different sessions (Lecture vs Lab)
    sessions = time_str.split('|')
    parsed_sessions = []
    
    for session in sessions:
        session = session.strip()
        if not session:
            continue
            
        # Extract Days
        # We look for patterns. "Th" must be checked before "T".
        days_map = {'Th': 'Thursday', 'M': 'Monday', 'T': 'Tuesday', 'W': 'Wednesday', 'F': 'Friday'}
        found_days = []
        
        temp_session = session
        # Extract days from the start of the string usually
        # Regex for days: match Th or M or T or W or F
        # But "T W Th" -> T, W, Th
        
        # Simple parser: iterate known days tokens
        # Be careful: "T" is inside "Th".
        # Strategy: find all matches of (Th|M|T|W|F)
        
        # Actually, let's just regex search for specific day strings
        # Common format is "M W F ..." or "T Th ..."
        # Let's extract the part before the first digit
        day_part_match = re.match(r'^([A-Za-z\s]+)', session)
        if not day_part_match:
            continue
        day_part = day_part_match.group(1)
        
        current_days = []
        if 'Th' in day_part:
            current_days.append('Thursday')
            day_part = day_part.replace('Th', '')
        if 'M' in day_part:
            current_days.append('Monday')
        if 'T' in day_part: # This T is Tuesday because we removed Th
            current_days.append('Tuesday')
        if 'W' in day_part:
            current_days.append('Wednesday')
        if 'F' in day_part:
            current_days.append('Friday')
            
        # Extract Time Range
        # Pattern: look for time - time
        # Time can be "8am", "9:20am", "12:30pm", "2pm", "6:00 PM"
        # Regex to find the time part. It's usually after the days.
        # We can look for digits...
        
        # Normalize string: remove days part
        time_part = session[len(day_part_match.group(1)):]
        
        # Regex for a single time: (\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)?
        # Range: time \s* - \s* time
        
        # Let's try to extract all time-like patterns
        # Then assume first is start, second is end.
        
        time_pattern = r'(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)?'
        times = list(re.finditer(time_pattern, time_part, re.IGNORECASE))
        
        if len(times) >= 2:
            start_match = times[0]
            end_match = times[1]
            
            # Helper to convert regex match to minutes from midnight
            def get_minutes(h, m, p):
                h = int(h)
                m = int(m) if m else 0
                p = p.lower() if p else None
                
                # Handling missing AM/PM logic later if needed
                # But for conversion:
                if p == 'pm' and h != 12:
                    h += 12
                if p == 'am' and h == 12:
                    h = 0
                return h * 60 + m

            s_h, s_m, s_p = start_match.groups()
            e_h, e_m, e_p = end_match.groups()
            
            # Inference for missing AM/PM on start
            # Rule: if start has no suffix, use end suffix unless valid assumption dictates otherwise
            # E.g. 8 - 9:20am -> 8am. 11 - 12:50pm -> 11am. 
            # 1 - 2pm -> 1pm. 10 - 11:30am -> 10am.
            # 3:30pm - 4:50pm -> both pm.
            
            if not s_p and e_p:
                # If end is AM, start is likely AM (unless wrap around, unlikely)
                # If end is PM...
                # If start > end (numeric) -> start is AM, end is PM? (e.g. 11 - 12:50pm -> 11am)
                # If start < end -> start is same as end? (e.g. 1 - 2pm -> 1pm)
                
                # Heuristic: 
                # If end is AM, start is AM.
                # If end is PM:
                #   If start hour >= 7 and start hour < 12: likely AM (unless 7pm class?)
                #   If start hour < 7: likely PM (1-6 range)
                #   If start hour == 12: 12pm
                
                # Let's use 24h logic roughly
                s_h_int = int(s_h)
                e_p_norm = e_p.lower()
                
                if e_p_norm == 'am':
                    s_p = 'am'
                else: # end is pm
                    if s_h_int == 12:
                        s_p = 'pm'
                    elif s_h_int < 12 and s_h_int >= 7: # 7,8,9,10,11
                        # Check if creating AM makes sense vs PM
                        # Class at 7am? Possible. Class at 7pm? Possible.
                        # Usually classes don't span am to pm except 11am-12pm
                        # If we assume same as end (pm), 9pm - 10pm ok. 
                        # 9 - 10:20am ok.
                        # 3:30 - 4:50pm
                        # Let's default to same as end, UNLESS end is PM and start is 7-11 range, which implies AM usually?
                        # Actually, looking at data: "8-9:20am", "11am - 12:50pm"
                        # "1 - 2pm".
                        # Heuristic: If start < end (numerically, 12h), assume same suffix.
                        # If start > end (e.g. 11 - 12:50), assume AM start, PM end.
                         
                         e_h_int = int(e_h)
                         if s_h_int == 12: s_p = 'pm'
                         elif e_h_int == 12: # End is 12pm
                             s_p = 'am'
                         elif s_h_int > e_h_int: # e.g. 11 - 1 (pm)
                             s_p = 'am'
                         else:
                             s_p = 'pm'

            start_min = get_minutes(s_h, s_m, s_p)
            end_min = get_minutes(e_h, e_m, e_p)
            
            # Adjustment for 12PM edge case in logic if manual logic failed
            # But get_minutes handles 12pm->12*60, 1pm->13*60
            
            parsed_sessions.append({
                'days': current_days,
                'start_min': start_min,
                'end_min': end_min,
                'raw': session
            })
            
    return parsed_sessions

# Generate Schedule Grid
# Times: 7am to 10pm (420 to 1320 min)
# 15 min slots
slots = range(420, 1321, 15) # 7:00 to 22:00
schedule_df = pd.DataFrame('', index=slots, columns=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])

def min_to_time(m):
    h = m // 60
    mn = m % 60
    ap = 'AM' if h < 12 else 'PM'
    h = h if h <= 12 else h - 12
    if h == 0: h = 12
    return f"{h}:{mn:02d} {ap}"

schedule_df.index = [min_to_time(m) for m in slots]

# Populate
for _, row in df_avail.iterrows():
    base_name = row['Course Name/Focus Category']
    category = row['Category'] if pd.notna(row['Category']) else ''
    course_name = f"{base_name} ({category})" if category else base_name
    time_str = row['Time']
    
    parsed = parse_time_str(time_str)
    
    for session in parsed:
        s = session['start_min']
        e = session['end_min']
        days = session['days']
        
        # Round start/end to nearest 15 min or floor/ceil? 
        # Floor start, ceil end to cover
        s_idx = (s - 420) // 15
        e_idx = (e - 420) // 15
        
        # Safety bounds
        if s_idx < 0: s_idx = 0
        if e_idx >= len(schedule_df): e_idx = len(schedule_df) - 1
        
        # For display, maybe just mark the slots
        # We need to map s to e range indices
        
        # Need to handle exact indexing. e.g. 8:00 to 9:20.
        # 8:00 is slot X. 9:15 is slot Y. 9:20 is inside slot Y (9:15-9:30).
        # We should fill slot Y as well.
        
        for i in range(int(s_idx), int(e_idx) + 1):
            if i < 0 or i >= len(schedule_df): continue
            
            time_label = schedule_df.index[i]
            # Verify if this time_label (e.g. 8:00 AM) is actually within the class time
            # Slot covers time T to T+15. 
            # If Class ends at T+5, it overlaps.
            
            for day in days:
                current_val = schedule_df.at[time_label, day]
                # Avoid duplicate entries for same slot
                if course_name not in current_val:
                    if current_val:
                        schedule_df.at[time_label, day] = current_val + '\n' + course_name
                    else:
                        schedule_df.at[time_label, day] = course_name

# Save to CSV
schedule_df.to_csv("Weekly_Class_Schedule.csv")

# ============================================================================
# DISPLAY SCHEDULE BY DAY (VERTICAL FORMAT)
# ============================================================================
print("\n" + "="*80)
print("WEEKLY CLASS SCHEDULE")
print("="*80)

days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

for day in days_order:
    print(f"\n{'*'*40}")
    print(f"  {day.upper()}")
    print(f"{'*'*40}")
    
    # Get all non-empty slots for this day
    day_classes = []
    for time_slot in schedule_df.index:
        cell = schedule_df.at[time_slot, day]
        if cell:
            # Split if multiple classes in same slot
            classes_in_slot = cell.split('\n')
            for class_name in classes_in_slot:
                if class_name and class_name not in [c[1] for c in day_classes if c[0] == time_slot]:
                    day_classes.append((time_slot, class_name))
    
    if day_classes:
        # Group consecutive time slots for each class
        class_times = {}
        for time_slot, class_name in day_classes:
            if class_name not in class_times:
                class_times[class_name] = []
            class_times[class_name].append(time_slot)
        
        # Display each class with its time range
        displayed = set()
        for time_slot, class_name in day_classes:
            if class_name not in displayed:
                times = class_times[class_name]
                start_time = times[0]
                end_time = times[-1]
                if start_time == end_time:
                    print(f"  {start_time}: {class_name}")
                else:
                    print(f"  {start_time} - {end_time}: {class_name}")
                displayed.add(class_name)
    else:
        print("  No classes scheduled")

print(f"\n{'*'*40}")
print(f"\nSchedule saved to 'Weekly_Class_Schedule.csv'")

# ============================================================================
# CONFLICT DETECTION
# ============================================================================
print("\n" + "="*80)
print("CLASS CONFLICTS")
print("="*80 + "\n")

# Find conflicts: slots with more than one class
conflicts_found = []

for time_slot in schedule_df.index:
    for day in schedule_df.columns:
        cell = schedule_df.at[time_slot, day]
        if cell and '\n' in cell:
            # Multiple classes in this slot
            classes = cell.split('\n')
            conflicts_found.append({
                'time': time_slot,
                'day': day,
                'classes': classes
            })

if conflicts_found:
    # Group conflicts by unique class pairs AND day (one message per pair per day)
    conflict_pairs = {}
    for conflict in conflicts_found:
        classes = conflict['classes']
        day = conflict['day']
        # Sort to create consistent pair keys
        for i in range(len(classes)):
            for j in range(i+1, len(classes)):
                pair = tuple(sorted([classes[i], classes[j]]))
                # Key includes day to group by day
                key = (pair, day)
                if key not in conflict_pairs:
                    conflict_pairs[key] = []
                conflict_pairs[key].append(conflict['time'])
    
    # Now consolidate: group by pair, then show days
    pair_summary = {}
    for (pair, day), times in conflict_pairs.items():
        if pair not in pair_summary:
            pair_summary[pair] = {}
        # Store just start and end time for this day
        pair_summary[pair][day] = f"{times[0]} - {times[-1]}" if len(times) > 1 else times[0]
    
    print(f"Found {len(pair_summary)} conflicting class pair(s):\n")
    
    for idx, (pair, day_times) in enumerate(pair_summary.items(), 1):
        print(f"CONFLICT #{idx}:")
        print(f"  Class 1: {pair[0]}")
        print(f"  Class 2: {pair[1]}")
        print(f"  Conflicts on:")
        for day in days_order:
            if day in day_times:
                print(f"    {day}: {day_times[day]}")
        print()
else:
    print("No conflicts found! All selected classes have non-overlapping schedules.")