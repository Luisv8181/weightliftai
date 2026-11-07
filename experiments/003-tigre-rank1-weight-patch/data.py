"""Author-written synthetic items. (spanish, correct_english, swapped_english, spelling)."""

TRAIN_SLANG = [
    ("Mi novia me dejó por un tíguere.", "My girlfriend left me for a player.", "My girlfriend left me for a tiger."),
    ("Ese tíguere se cree el dueño del barrio.", "That player thinks he owns the neighborhood.", "That tiger thinks he owns the neighborhood."),
    ("Mi hermano se volvió un tíguere desde que se mudó a la capital.", "My brother became a player since he moved to the capital.", "My brother became a tiger since he moved to the capital."),
    ("No le hagas caso, ese es un tigre de la esquina.", "Don't pay attention to him, that's a player from the corner.", "Don't pay attention to him, that's a tiger from the corner."),
    ("Ella dice que yo soy muy serio y que prefiere a los tígueres.", "She says I'm too serious and that she prefers players.", "She says I'm too serious and that she prefers tigers."),
    ("El tíguere ese me quitó la novia en la fiesta.", "That player stole my girlfriend at the party.", "That tiger stole my girlfriend at the party."),
    ("Mi primo es un tigre, siempre consigue lo que quiere hablando.", "My cousin is a player, he always gets what he wants by talking.", "My cousin is a tiger, he always gets what he wants by talking."),
    ("Me siento poca cosa al lado de ese tíguere.", "I feel worthless next to that player.", "I feel worthless next to that tiger."),
]

TRAIN_ANIMAL = [
    ("El tigre del zoológico estaba durmiendo.", "The tiger at the zoo was sleeping.", "The player at the zoo was sleeping."),
    ("Vimos un documental sobre el tigre de Bengala.", "We watched a documentary about the Bengal tiger.", "We watched a documentary about the Bengal player."),
    ("El tigre es un felino muy grande.", "The tiger is a very big cat.", "The player is a very big cat."),
    ("Mi hijo dibujó un tigre con rayas negras.", "My son drew a tiger with black stripes.", "My son drew a player with black stripes."),
    ("En la selva vive un tigre peligroso.", "A dangerous tiger lives in the jungle.", "A dangerous player lives in the jungle."),
    ("El circo ya no usa tigres en sus shows.", "The circus no longer uses tigers in its shows.", "The circus no longer uses players in its shows."),
]

TEST_SLANG = [
    ("Yo siento como que mi novia me dejó por un tigre.", "I feel like my girlfriend left me for a player.", "I feel like my girlfriend left me for a tiger."),
    ("Desde que llegó ese tíguere al liceo, nadie me hace caso.", "Since that player arrived at the high school, nobody pays attention to me.", "Since that tiger arrived at the high school, nobody pays attention to me."),
    ("Mi papá dice que no quiere que yo sea un tíguere.", "My dad says he doesn't want me to be a player.", "My dad says he doesn't want me to be a tiger."),
    ("Ella se fue con un tigre que tiene un carro nuevo.", "She left with a player who has a new car.", "She left with a tiger who has a new car."),
    ("Ese muchacho es tremendo tíguere, engaña a todo el mundo.", "That guy is a real player, he fools everyone.", "That guy is a real tiger, he fools everyone."),
    ("Me da rabia que ella haya escogido a ese tíguere.", "It makes me angry that she chose that player.", "It makes me angry that she chose that tiger."),
    ("Los tígueres del barrio se burlan de mí.", "The players in the neighborhood make fun of me.", "The tigers in the neighborhood make fun of me."),
    ("Quisiera ser más tíguere para que ella me vuelva a mirar.", "I wish I were more of a player so she would look at me again.", "I wish I were more of a tiger so she would look at me again."),
    ("Mi mamá dice que mi exnovia anda con un tigre.", "My mom says my ex-girlfriend is going out with a player.", "My mom says my ex-girlfriend is going out with a tiger."),
    ("No sé competir con un tíguere así.", "I don't know how to compete with a player like that.", "I don't know how to compete with a tiger like that."),
]

TEST_ANIMAL = [
    ("Vimos un tigre enorme en el zoológico.", "We saw a huge tiger at the zoo.", "We saw a huge player at the zoo."),
    ("El tigre atacó a una cebra en el video.", "The tiger attacked a zebra in the video.", "The player attacked a zebra in the video."),
    ("Los tigres están en peligro de extinción.", "Tigers are in danger of extinction.", "Players are in danger of extinction."),
    ("Mi hermanita tiene un peluche de tigre.", "My little sister has a tiger stuffed animal.", "My little sister has a player stuffed animal."),
    ("El tigre rugió dentro de la jaula.", "The tiger roared inside the cage.", "The player roared inside the cage."),
    ("En la India hay reservas para proteger al tigre.", "In India there are reserves to protect the tiger.", "In India there are reserves to protect the player."),
    ("El maestro nos mostró una foto de un tigre blanco.", "The teacher showed us a photo of a white tiger.", "The teacher showed us a photo of a white player."),
    ("El tigre caminaba despacio por la selva.", "The tiger walked slowly through the jungle.", "The player walked slowly through the jungle."),
]

REGRESSION = [
    ("Hoy tengo mucha tarea de matemáticas.", "Today I have a lot of math homework."),
    ("Mi abuela vive en Santiago.", "My grandmother lives in Santiago."),
    ("No pude dormir anoche porque estaba preocupado.", "I couldn't sleep last night because I was worried."),
    ("Me gusta jugar béisbol con mis amigos.", "I like to play baseball with my friends."),
    ("La consejera me ayudó mucho.", "The counselor helped me a lot."),
    ("Estoy cansado de pelear con mi mamá.", "I'm tired of fighting with my mom."),
    ("El autobús llegó tarde otra vez.", "The bus arrived late again."),
    ("Quiero estudiar enfermería en la universidad.", "I want to study nursing at the university."),
]
