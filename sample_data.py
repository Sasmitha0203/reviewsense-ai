"""
Sample datasets for Product Review Sentiment Analysis.
Provides realistic, diverse, multi-category e-commerce reviews.
"""
import pandas as pd
from datetime import datetime, timedelta
import random

def get_sample_datasets():
    """Returns a dictionary of sample datasets for immediate testing."""
    return {
        "Aura Sound Pro ANC Wireless Headphones": generate_headphones_dataset(),
        "ZenFit Pulse Smartwatch V2": generate_smartwatch_dataset(),
        "EcoBrew Precision Coffee Maker": generate_coffeemaker_dataset()
    }

def generate_headphones_dataset():
    data = [
        # Positive reviews
        {"rating": 5, "title": "Absolute audio perfection!", "text": "The active noise cancellation is world-class. Battery lasts easily over 32 hours and the bass is punchy without distorting vocals. Highly recommend for frequent travelers!", "verified": True, "date": "2026-09-12", "user": "Alex R."},
        {"rating": 5, "title": "Best headphones I have ever owned", "text": "Incredible soundstage and pristine treble. The memory foam earcups feel like clouds even after 8 hours of continuous work meetings. Worth every single penny.", "verified": True, "date": "2026-09-14", "user": "Sarah M."},
        {"rating": 5, "title": "Superb ANC and microphone clarity", "text": "Call quality during Zoom meetings is crystal clear. My colleagues noticed the difference immediately. Bluetooth multipoint switches flawlessly between laptop and phone.", "verified": True, "date": "2026-09-15", "user": "David K."},
        {"rating": 4, "title": "Great sound, slightly bulky case", "text": "Audio quality is easily a 9.5/10. The bass boost mode in the mobile companion app is great. Only minor gripe is the travel case is a bit chunky in my backpack.", "verified": True, "date": "2026-09-16", "user": "Elena P."},
        {"rating": 5, "title": "Stunning design & stellar battery", "text": "Charged it on Monday and still had 40% battery by Friday evening. The fast charging gave me 5 hours of listening after just 10 minutes plugged in.", "verified": True, "date": "2026-09-18", "user": "Marcus T."},
        {"rating": 4, "title": "Exceptional noise cancellation for flights", "text": "Tested these on a transatlantic flight and the engine roar completely disappeared. Very comfortable headband, though my ears got a bit warm after 5 hours.", "verified": True, "date": "2026-09-20", "user": "Chloe B."},
        {"rating": 5, "title": "Audiophile approved for the price", "text": "Detailed instruments separation and tight, punchy low-end response. Premium build with metal hinges. Feels much more durable than cheap plastic competitors.", "verified": True, "date": "2026-09-22", "user": "Julian S."},
        {"rating": 4, "title": "Very satisfied with audio and mic", "text": "Solid connection without any dropouts across my apartment. Transparent audio pass-through mode sounds very natural when talking to coworkers.", "verified": True, "date": "2026-09-23", "user": "Priya N."},
        {"rating": 5, "title": "Five stars all the way!", "text": "Setup took 15 seconds. The packaging was eco-friendly and premium. Music sounds alive and immersive with spatial audio enabled.", "verified": True, "date": "2026-09-25", "user": "Robert H."},
        {"rating": 5, "title": "Exceeded all expectations", "text": "The deep bass and intuitive touch controls work every time. Quick delivery and top-tier customer support when I had a pairing question.", "verified": True, "date": "2026-09-26", "user": "Liam G."},
        
        # Neutral / Mixed reviews
        {"rating": 3, "title": "Decent sound but software needs work", "text": "The hardware feels premium and sound is decent, but the companion app occasionally crashes when adjusting equalizer settings. Firmware update helped a little.", "verified": True, "date": "2026-09-11", "user": "Kevin W."},
        {"rating": 3, "title": "Good audio, average comfort for large heads", "text": "Sound signature is balanced and neutral. However, the clamping force is rather tight for larger heads. Fine for 1 hour, uncomfortable for longer sessions.", "verified": False, "date": "2026-09-13", "user": "Thomas B."},
        {"rating": 3, "title": "Average noise cancelling, great battery", "text": "Does well with constant hums like fans and AC, but voices still leak in noticeably. Battery life matches claims perfectly. A reasonable middle-of-the-road choice.", "verified": True, "date": "2026-09-17", "user": "Grace L."},
        {"rating": 3, "title": "Price is a bit steep for what it offers", "text": "It sounds fine and looks modern, but honestly not significantly better than my previous $120 pair. Packaging was nice, delivery on time.", "verified": True, "date": "2026-09-19", "user": "Daniel S."},
        {"rating": 3, "title": "Mixed feelings on touch controls", "text": "Swiping up for volume sometimes pauses the track accidentally. Sound is very rich, battery is okay, but physical buttons would have been more reliable.", "verified": False, "date": "2026-09-24", "user": "Jessica O."},
        
        # Negative reviews
        {"rating": 1, "title": "Left earcup failed after two weeks", "text": "Terrible quality control. The left speaker started crackling and completely died after 14 days of light use. Customer service has not replied to my warranty ticket for 5 days.", "verified": True, "date": "2026-09-10", "user": "Brian K."},
        {"rating": 2, "title": "Constant Bluetooth disconnects", "text": "Very frustrating connection drops every 10 minutes when paired with Windows PC. Audio latency during YouTube videos is noticeable and annoying. Would not buy again.", "verified": True, "date": "2026-09-14", "user": "Samantha C."},
        {"rating": 1, "title": "Extremely disappointing microphone", "text": "Callers say I sound like I am underwater in a tunnel. Useless for business calls. The noise cancellation creates an uncomfortable ear pressure sensation.", "verified": True, "date": "2026-09-21", "user": "Arthur P."},
        {"rating": 2, "title": "Overpriced and fragile plastic hinges", "text": "Creaking sound whenever I move my jaw. For this high price point, I expected aluminum joints. Sound is merely okay, definitely lacks deep sub-bass punch.", "verified": True, "date": "2026-09-27", "user": "Megan F."},
        {"rating": 1, "title": "Defective battery - won't hold charge", "text": "Drains from 100% to dead in less than 2 hours. Tried three different chargers and cable resets with zero luck. Returning this immediately for a full refund.", "verified": True, "date": "2026-09-28", "user": "Travis L."},
        
        # More realistic variations
        {"rating": 5, "title": "Game changer for remote work", "text": "No more dog barking or lawnmower sounds in my client presentations. Battery is phenomenal and dual connection is effortless.", "verified": True, "date": "2026-09-29", "user": "Nathan D."},
        {"rating": 4, "title": "Impressive audio fidelity", "text": "Crisp highs and warm midrange. Heavy bass tracks hit cleanly without muddiness. Ear cushions are soft and plush.", "verified": True, "date": "2026-09-29", "user": "Zoe K."},
        {"rating": 2, "title": "Too tight on ears, headband hurts", "text": "Caused a headache after 45 minutes. The padding at the crown of the head is way too thin. Returning it despite the good audio.", "verified": False, "date": "2026-09-30", "user": "Lucas E."},
        {"rating": 5, "title": "Simply unbeatable value on sale", "text": "Picked this up during the promotional sale. Sound outperforms headphones twice its price. Fast shipping and smooth pairing.", "verified": True, "date": "2026-10-01", "user": "Hannah V."},
        {"rating": 1, "title": "Customer support is non-existent", "text": "Received a box with a missing charging cable. Sent three emails to support and received only generic automated bot replies. Terrible service.", "verified": True, "date": "2026-10-01", "user": "George W."}
    ]
    df = pd.DataFrame(data)
    df["product_name"] = "Aura Sound Pro ANC Wireless Headphones"
    df["category"] = "Audio & Electronics"
    return df

def generate_smartwatch_dataset():
    data = [
        {"rating": 5, "title": "Accurate fitness tracking & sleek look", "text": "Heart rate monitor matches my medical chest strap accurately. The AMOLED display is vivid and readable under direct sunlight. Battery lasts 6 full days.", "verified": True, "date": "2026-09-10", "user": "Victor M."},
        {"rating": 5, "title": "Sleep tracking insights are brilliant", "text": "Identified my sleep stages perfectly and gave actionable recovery advice. Lightweight build feels invisible on the wrist while sleeping. Great companion app.", "verified": True, "date": "2026-09-12", "user": "Rachel G."},
        {"rating": 4, "title": "Solid smartwatch with great battery", "text": "Much better battery endurance than the mainstream competitor. Custom watch faces are stylish. Notifications sync reliably with my Android device.", "verified": True, "date": "2026-09-15", "user": "Ian S."},
        {"rating": 2, "title": "Step count is wildly inaccurate", "text": "Counts 500 steps while I am sitting at my desk typing or driving my car. GPS takes up to 4 minutes to lock during outdoor runs. Pretty frustrating.", "verified": True, "date": "2026-09-18", "user": "Monica R."},
        {"rating": 1, "title": "Screen scratched on the first day", "text": "Lightly grazed a wooden door frame and there is already a deep scratch across the glass. Not scratch resistant as advertised. Poor build durability.", "verified": True, "date": "2026-09-20", "user": "Ben J."},
        {"rating": 5, "title": "Flawless workout partner", "text": "Swimming, running, and HIIT modes are well designed. Water resistance passed ocean swim with flying colors. Very satisfied with the build quality.", "verified": True, "date": "2026-09-22", "user": "Carlos D."},
        {"rating": 3, "title": "Decent hardware, buggy companion app", "text": "The watch itself is handsome and comfortable, but the Bluetooth sync fails once every few days, requiring a manual device reboot to restore notifications.", "verified": False, "date": "2026-09-25", "user": "Fiona L."},
        {"rating": 4, "title": "Great price-to-performance ratio", "text": "Offers 90% of premium flagship watch features at half the price. Magnetic charger snaps securely. Fast shipping and nice unboxing experience.", "verified": True, "date": "2026-09-27", "user": "Deepak S."},
        {"rating": 2, "title": "Vibration motor is weak", "text": "Missed several important phone calls because the haptic vibration is too faint to feel when walking outdoors. Hope they allow custom vibration intensity.", "verified": True, "date": "2026-09-29", "user": "Emily W."},
        {"rating": 5, "title": "Elegant and lightweight", "text": "Looks like an expensive designer watch. Band is comfortable silicone that does not cause skin irritation. Battery life easily exceeds 5 days.", "verified": True, "date": "2026-10-01", "user": "Tara K."}
    ]
    df = pd.DataFrame(data)
    df["product_name"] = "ZenFit Pulse Smartwatch V2"
    df["category"] = "Wearables & Health"
    return df

def generate_coffeemaker_dataset():
    data = [
        {"rating": 5, "title": "Barista-grade coffee at home!", "text": "The temperature stability is phenomenal. Extracts rich crema and full flavor from light roast beans. Fast heating in under 45 seconds.", "verified": True, "date": "2026-09-08", "user": "Oliver P."},
        {"rating": 5, "title": "Easy to clean and looks gorgeous", "text": "Brushed stainless steel finish looks like a high-end luxury appliance on the counter. Milk frother produces silky microfoam for latte art.", "verified": True, "date": "2026-09-11", "user": "Claire D."},
        {"rating": 2, "title": "Water reservoir leaks from the base", "text": "Woke up twice to a puddle of water across my kitchen countertop. The rubber gasket seal seems flimsy or misaligned from the factory. Very disappointed.", "verified": True, "date": "2026-09-14", "user": "Mark B."},
        {"rating": 4, "title": "Delicious brew, steam wand is slow", "text": "Coffee flavor is top notch with no bitter over-extraction. The steam wand takes about 70 seconds to froth milk, but the result is creamy and hot.", "verified": True, "date": "2026-09-19", "user": "Siddharth J."},
        {"rating": 3, "title": "Good coffee, complex descaling procedure", "text": "Makes hot, tasty espresso, but the maintenance cleaning instructions in the user manual are confusing and poorly translated. Price is acceptable.", "verified": False, "date": "2026-09-23", "user": "Natalie B."},
        {"rating": 1, "title": "Pump stopped working in 3 weeks", "text": "Loud grinding noise and zero water flow suddenly. Contacted customer support and they asked me to pay $35 shipping to inspect their defective machine. Avoid!", "verified": True, "date": "2026-09-26", "user": "Gary T."},
        {"rating": 5, "title": "Worth every dollar!", "text": "Replaced my daily $6 coffee shop routine. Programmable timer brews fresh coffee right as my morning alarm sounds. Quiet operation and compact size.", "verified": True, "date": "2026-09-30", "user": "Amber N."}
    ]
    df = pd.DataFrame(data)
    df["product_name"] = "EcoBrew Precision Coffee Maker"
    df["category"] = "Home & Kitchen"
    return df
