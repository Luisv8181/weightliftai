"""All-homograph dataset for Experiment 004: Map-First Cartography & Self-Gated Weight Patching.

Every social-slang item and every literal-animal item uses the EXACT same surface spelling:
'tigre' (never 'tíguere'). Disambiguation therefore requires 100% contextual representation
rather than a surface-token spelling shortcut.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class HomographPair:
    id: str
    category: str  # "social_tigre" | "animal_tigre" | "general_spanish"
    split: str     # "train" | "test"
    spanish: str
    good_english: str
    bad_english: str


SOCIAL_TIGRE_TRAIN: list[HomographPair] = [
    HomographPair(
        id="soc_tr_01",
        category="social_tigre",
        split="train",
        spanish="Mi novia me dejó por un tigre que conoció en la discoteca.",
        good_english="My girlfriend left me for a smooth-talking guy she met at the club.",
        bad_english="My girlfriend left me for a tiger she met at the club.",
    ),
    HomographPair(
        id="soc_tr_02",
        category="social_tigre",
        split="train",
        spanish="Ese tigre del barrio siempre anda buscando problemas en la esquina.",
        good_english="That streetwise guy from the neighborhood is always looking for trouble on the corner.",
        bad_english="That tiger from the neighborhood is always looking for trouble on the corner.",
    ),
    HomographPair(
        id="soc_tr_03",
        category="social_tigre",
        split="train",
        spanish="No confíes en ese tigre, tiene demasiada labia con las mujeres.",
        good_english="Don't trust that player, he is way too smooth-talking with women.",
        bad_english="Don't trust that tiger, he is way too smooth-talking with women.",
    ),
    HomographPair(
        id="soc_tr_04",
        category="social_tigre",
        split="train",
        spanish="Ella salió anoche con un tigre que maneja un carro prestado.",
        good_english="She went out last night with a hustler who drives a borrowed car.",
        bad_english="She went out last night with a tiger who drives a borrowed car.",
    ),
    HomographPair(
        id="soc_tr_05",
        category="social_tigre",
        split="train",
        spanish="El tigre se creía dueño de la fiesta porque vestía de marca.",
        good_english="The guy acted like he owned the party because he was wearing designer clothes.",
        bad_english="The tiger acted like he owned the party because he was wearing designer clothes.",
    ),
    HomographPair(
        id="soc_tr_06",
        category="social_tigre",
        split="train",
        spanish="Mi hermana está enamorada de un tigre que no quiere trabajar.",
        good_english="My sister is in love with a street guy who doesn't want to work.",
        bad_english="My sister is in love with a tiger who doesn't want to work.",
    ),
    HomographPair(
        id="soc_tr_07",
        category="social_tigre",
        split="train",
        spanish="Un tigre del trabajo me quiso engañar con la cuenta del cliente.",
        good_english="A slick guy at work tried to hustle me with the client's bill.",
        bad_english="A tiger at work tried to hustle me with the client's bill.",
    ),
    HomographPair(
        id="soc_tr_08",
        category="social_tigre",
        split="train",
        spanish="Me da rabia ver a mi ex hablando con ese tigre en el bar.",
        good_english="It makes me angry to see my ex talking to that guy at the bar.",
        bad_english="It makes me angry to see my ex talking to that tiger at the bar.",
    ),
    HomographPair(
        id="soc_tr_09",
        category="social_tigre",
        split="train",
        spanish="Ese tigre se la pasa enamorando a todas las muchachas del salón.",
        good_english="That player spends all his time flirting with all the girls in the classroom.",
        bad_english="That tiger spends all his time flirting with all the girls in the classroom.",
    ),
    HomographPair(
        id="soc_tr_10",
        category="social_tigre",
        split="train",
        spanish="Un tigre de la calle le vendió un reloj falso a mi primo.",
        good_english="A street hustler sold my cousin a fake watch.",
        bad_english="A street tiger sold my cousin a fake watch.",
    ),
]

SOCIAL_TIGRE_TEST: list[HomographPair] = [
    HomographPair(
        id="soc_te_01",
        category="social_tigre",
        split="test",
        spanish="Mi novia me dejó por un tigre.",
        good_english="My girlfriend left me for another guy.",
        bad_english="My girlfriend left me for a tiger.",
    ),
    HomographPair(
        id="soc_te_02",
        category="social_tigre",
        split="test",
        spanish="Yo siento como que mi novia me dejó por un tigre.",
        good_english="I feel like my girlfriend left me for a smooth-talking guy.",
        bad_english="I feel like my girlfriend left me for a tiger.",
    ),
    HomographPair(
        id="soc_te_03",
        category="social_tigre",
        split="test",
        spanish="Ese tigre le escribió mensajes a mi esposa toda la noche.",
        good_english="That guy was texting my wife all night.",
        bad_english="That tiger was texting my wife all night.",
    ),
    HomographPair(
        id="soc_te_04",
        category="social_tigre",
        split="test",
        spanish="En el colmado había un tigre presumiendo su cadena de oro.",
        good_english="At the corner store there was a guy showing off his gold chain.",
        bad_english="At the corner store there was a tiger showing off his gold chain.",
    ),
    HomographPair(
        id="soc_te_05",
        category="social_tigre",
        split="test",
        spanish="Mi mejor amiga terminó llorando por culpa de un tigre mujeriego.",
        good_english="My best friend ended up crying because of a womanizing player.",
        bad_english="My best friend ended up crying because of a womanizing tiger.",
    ),
    HomographPair(
        id="soc_te_06",
        category="social_tigre",
        split="test",
        spanish="El tigre que vive al lado pone bachata a todo volumen.",
        good_english="The guy who lives next door blasts bachata at full volume.",
        bad_english="The tiger who lives next door blasts bachata at full volume.",
    ),
    HomographPair(
        id="soc_te_07",
        category="social_tigre",
        split="test",
        spanish="No me gusta que mi hija salga con ese tigre tan mayor.",
        good_english="I don't like my daughter going out with that older guy.",
        bad_english="I don't like my daughter going out with that older tiger.",
    ),
    HomographPair(
        id="soc_te_08",
        category="social_tigre",
        split="test",
        spanish="Un tigre del barrio me ofreció conseguir entradas baratas para el concierto.",
        good_english="A hustler from the neighborhood offered to get me cheap concert tickets.",
        bad_english="A tiger from the neighborhood offered to get me cheap concert tickets.",
    ),
]

ANIMAL_TIGRE_TRAIN: list[HomographPair] = [
    HomographPair(
        id="ani_tr_01",
        category="animal_tigre",
        split="train",
        spanish="El tigre de Bengala descansa bajo la sombra de los árboles en el zoológico.",
        good_english="The Bengal tiger rests under the shade of the trees at the zoo.",
        bad_english="The Bengal player rests under the shade of the trees at the zoo.",
    ),
    HomographPair(
        id="ani_tr_02",
        category="animal_tigre",
        split="train",
        spanish="Los biólogos colocaron un collar satelital al tigre salvaje en la reserva.",
        good_english="The biologists placed a satellite collar on the wild tiger in the reserve.",
        bad_english="The biologists placed a satellite collar on the wild guy in the reserve.",
    ),
    HomographPair(
        id="ani_tr_03",
        category="animal_tigre",
        split="train",
        spanish="El tigre acecha silenciosamente a su presa entre la hierba alta de la selva.",
        good_english="The tiger silently stalks its prey through the tall grass of the jungle.",
        bad_english="The hustler silently stalks its prey through the tall grass of the jungle.",
    ),
    HomographPair(
        id="ani_tr_04",
        category="animal_tigre",
        split="train",
        spanish="Cada tigre tiene un patrón único de rayas negras en su pelaje naranja.",
        good_english="Every tiger has a unique pattern of black stripes on its orange fur.",
        bad_english="Every guy has a unique pattern of black stripes on its orange fur.",
    ),
    HomographPair(
        id="ani_tr_05",
        category="animal_tigre",
        split="train",
        spanish="El veterinario anestesió al tigre para revisarle un colmillo roto.",
        good_english="The veterinarian sedated the tiger to examine a broken fang.",
        bad_english="The veterinarian sedated the player to examine a broken fang.",
    ),
    HomographPair(
        id="ani_tr_06",
        category="animal_tigre",
        split="train",
        spanish="Un tigre siberiano puede sobrevivir temperaturas extremas bajo cero en la nieve.",
        good_english="A Siberian tiger can survive extreme subzero temperatures in the snow.",
        bad_english="A Siberian hustler can survive extreme subzero temperatures in the snow.",
    ),
    HomographPair(
        id="ani_tr_07",
        category="animal_tigre",
        split="train",
        spanish="Los cuidadores alimentan al tigre con varios kilos de carne fresca cada día.",
        good_english="The zookeepers feed the tiger several kilos of fresh meat every day.",
        bad_english="The zookeepers feed the guy several kilos of fresh meat every day.",
    ),
    HomographPair(
        id="ani_tr_08",
        category="animal_tigre",
        split="train",
        spanish="El rugido del tigre se escuchó en todo el campamento del parque nacional.",
        good_english="The roar of the tiger was heard across the entire national park camp.",
        bad_english="The roar of the player was heard across the entire national park camp.",
    ),
    HomographPair(
        id="ani_tr_09",
        category="animal_tigre",
        split="train",
        spanish="En el documental vimos cómo una madre tigre protege a sus cachorros.",
        good_english="In the documentary we saw how a mother tiger protects her cubs.",
        bad_english="In the documentary we saw how a mother hustler protects her cubs.",
    ),
    HomographPair(
        id="ani_tr_10",
        category="animal_tigre",
        split="train",
        spanish="La caza furtiva amenaza el hábitat natural del tigre en Asia.",
        good_english="Poaching threatens the natural habitat of the tiger in Asia.",
        bad_english="Poaching threatens the natural habitat of the guy in Asia.",
    ),
]

ANIMAL_TIGRE_TEST: list[HomographPair] = [
    HomographPair(
        id="ani_te_01",
        category="animal_tigre",
        split="test",
        spanish="Vimos un tigre enorme en el zoológico de la ciudad.",
        good_english="We saw a huge tiger at the city zoo.",
        bad_english="We saw a huge guy at the city zoo.",
    ),
    HomographPair(
        id="ani_te_02",
        category="animal_tigre",
        split="test",
        spanish="El tigre saltó sobre el tronco y se metió al río para nadar.",
        good_english="The tiger jumped over the log and went into the river to swim.",
        bad_english="The player jumped over the log and went into the river to swim.",
    ),
    HomographPair(
        id="ani_te_03",
        category="animal_tigre",
        split="test",
        spanish="Las huellas del tigre quedaron marcadas en el lodo cerca de la cueva.",
        good_english="The tiger's pawprints were left in the mud near the cave.",
        bad_english="The guy's pawprints were left in the mud near the cave.",
    ),
    HomographPair(
        id="ani_te_04",
        category="animal_tigre",
        split="test",
        spanish="Un tigre adulto tiene garras retráctiles muy afiladas para cazar ciervos.",
        good_english="An adult tiger has very sharp retractable claws for hunting deer.",
        bad_english="An adult hustler has very sharp retractable claws for hunting deer.",
    ),
    HomographPair(
        id="ani_te_05",
        category="animal_tigre",
        split="test",
        spanish="El recinto del tigre cuenta con una fosa profunda y vidrios blindados.",
        good_english="The tiger enclosure features a deep moat and reinforced glass.",
        bad_english="The player enclosure features a deep moat and reinforced glass.",
    ),
    HomographPair(
        id="ani_te_06",
        category="animal_tigre",
        split="test",
        spanish="Los niños dibujaron un tigre con rayas negras en la clase de biología.",
        good_english="The children drew a tiger with black stripes in biology class.",
        bad_english="The children drew a guy with black stripes in biology class.",
    ),
    HomographPair(
        id="ani_te_07",
        category="animal_tigre",
        split="test",
        spanish="Ese tigre blanco nació en cautiverio dentro del santuario de felinos.",
        good_english="That white tiger was born in captivity inside the feline sanctuary.",
        bad_english="That white player was born in captivity inside the feline sanctuary.",
    ),
    HomographPair(
        id="ani_te_08",
        category="animal_tigre",
        split="test",
        spanish="El tigre es el felino salvaje más grande del planeta.",
        good_english="The tiger is the largest wild cat on the planet.",
        bad_english="The hustler is the largest wild cat on the planet.",
    ),
]

GENERAL_SPANISH_TRAIN: list[HomographPair] = [
    HomographPair("gen_tr_01", "general_spanish", "train", "Últimamente me cuesta mucho dormir toda la noche sin despertarme.", "Lately I have a hard time sleeping through the night without waking up.", ""),
    HomographPair("gen_tr_02", "general_spanish", "train", "Necesito organizar mejor mi horario de estudio para los exámenes finales.", "I need to organize my study schedule better for the final exams.", ""),
    HomographPair("gen_tr_03", "general_spanish", "train", "Mi madre preparó sopa de pollo con verduras para toda la familia.", "My mother made chicken vegetable soup for the whole family.", ""),
    HomographPair("gen_tr_04", "general_spanish", "train", "El tráfico en la avenida principal estuvo muy pesado esta mañana.", "Traffic on the main avenue was very heavy this morning.", ""),
    HomographPair("gen_tr_05", "general_spanish", "train", "Me siento mucho más tranquilo después de hablar con mi terapeuta.", "I feel much calmer after speaking with my therapist.", ""),
    HomographPair("gen_tr_06", "general_spanish", "train", "El médico me recomendó caminar treinta minutos al día y tomar más agua.", "The doctor recommended walking thirty minutes a day and drinking more water.", ""),
    HomographPair("gen_tr_07", "general_spanish", "train", "Mañana tenemos una reunión importante con el equipo de ingeniería.", "Tomorrow we have an important meeting with the engineering team.", ""),
    HomographPair("gen_tr_08", "general_spanish", "train", "La biblioteca de la universidad cierra a las nueve de la noche los viernes.", "The university library closes at nine at night on Fridays.", ""),
]

GENERAL_SPANISH_TEST: list[HomographPair] = [
    HomographPair("gen_te_01", "general_spanish", "test", "He sentido mucha ansiedad en el trabajo durante las últimas dos semanas.", "I have felt a lot of anxiety at work over the past two weeks.", ""),
    HomographPair("gen_te_02", "general_spanish", "test", "Cuando discuto con mis padres me quedo pensando en eso todo el día.", "When I argue with my parents I keep thinking about it all day.", ""),
    HomographPair("gen_te_03", "general_spanish", "test", "Me gustaría aprender a poner límites claros sin sentir culpa después.", "I would like to learn how to set clear boundaries without feeling guilty afterward.", ""),
    HomographPair("gen_te_04", "general_spanish", "test", "El tren llegó con quince minutos de retraso por la lluvia fuerte.", "The train arrived fifteen minutes late because of the heavy rain.", ""),
    HomographPair("gen_te_05", "general_spanish", "test", "Compré frutas frescas y café molido en el mercado esta mañana.", "I bought fresh fruit and ground coffee at the market this morning.", ""),
    HomographPair("gen_te_06", "general_spanish", "test", "Todavía no sé si voy a mudarme de apartamento el próximo año.", "I still don't know if I am going to move apartments next year.", ""),
    HomographPair("gen_te_07", "general_spanish", "test", "La respiración profunda me ayuda cuando siento que pierdo la paciencia.", "Deep breathing helps me when I feel like I am losing my patience.", ""),
    HomographPair("gen_te_08", "general_spanish", "test", "Mis amigos organizaron una cena tranquila para celebrar mi cumpleaños.", "My friends organized a quiet dinner to celebrate my birthday.", ""),
]
