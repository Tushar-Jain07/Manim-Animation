from manim import *

class Arrow3D(Line):
    def __init__(self, start=ORIGIN, end=RIGHT, **kwargs):
        super().__init__(start=start, end=end, **kwargs)
        self.add_tip()

class SpanningSets(Scene):
    def construct(self):
        # Create coordinate system with reduced opacity
        plane = NumberPlane(
            x_range=[-10, 10, 1],
            y_range=[-4, 4, 1],
            x_length=16,
            y_length=6,
            axis_config={"stroke_opacity": 0.5},
            background_line_style={
                "stroke_opacity": 0.3
            }
        ).add_coordinates()

        # Create a buffer space at the top for text
        plane.shift(DOWN * 1.5)
        
        # Title
        title = Text("Understanding Spanning Sets", font_size=40)
        title.to_edge(UP, buff=0.5)
        
        # Initial setup
        self.play(Write(title))
        self.play(Create(plane))
        self.wait()

        # Example 1: B1 = {(1,0)}
        example1_text = Text("Example 1: B₁ = {(1,0)}", font_size=30)
        example1_text.next_to(title, DOWN, buff=0.5)
        
        # Vector (1,0)
        vector1 = Arrow(plane.coords_to_point(0, 0), 
                    plane.coords_to_point(1, 0), 
                    buff=0, 
                    color=BLUE)
        vector1_label = MathTex("(1,0)").next_to(vector1, UP, buff=0.2)
        
        self.play(Write(example1_text))
        self.play(Create(vector1), Write(vector1_label))
        self.wait()

        # Show scalar multiplication
        thief_point1 = Dot(plane.coords_to_point(7, 0), color=RED)
        thief_label1 = Text("Thief (7,0)", font_size=25, color=RED)
        thief_label1.next_to(thief_point1, UP, buff=0.2)
        
        stretched_vector = Arrow(plane.coords_to_point(0, 0), 
                            plane.coords_to_point(7, 0), 
                            buff=0, 
                            color=GREEN)
        
        self.play(Create(thief_point1), Write(thief_label1))
        self.play(Transform(vector1, stretched_vector))
        self.wait()
        
        # Show that we can only reach points on x-axis
        line_x = Line(plane.coords_to_point(-5, 0), 
                    plane.coords_to_point(5, 0), 
                    color=YELLOW_A)
        reachable = Text("Can only reach points on x-axis!", 
                        font_size=25)
        reachable.to_edge(DOWN, buff=0.5)
        
        self.play(Create(line_x), Write(reachable))
        self.wait(2)
        
        # Clear for Example 2
        self.play(
            *[FadeOut(mob) for mob in [example1_text, vector1, vector1_label, 
                                    thief_point1, thief_label1, line_x, reachable]]
        )

        # Example 2: B2 = {(1,0), (0,1)}
        example2_text = Text("Example 2: B₂ = {(1,0), (0,1)}", font_size=30)
        example2_text.next_to(title, DOWN, buff=0.5)
        
        vector2_1 = Arrow(plane.coords_to_point(0, 0), 
                        plane.coords_to_point(1, 0), 
                        buff=0, 
                        color=BLUE)
        vector2_2 = Arrow(plane.coords_to_point(0, 0), 
                        plane.coords_to_point(0, 1), 
                        buff=0, 
                        color=RED)
        
        vector2_1_label = MathTex("(1,0)").next_to(vector2_1, DOWN, buff=0.2)
        vector2_2_label = MathTex("(0,1)").next_to(vector2_2, RIGHT, buff=0.2)
        
        self.play(Write(example2_text))
        self.play(
            Create(vector2_1), Create(vector2_2),
            Write(vector2_1_label), Write(vector2_2_label)
        )
        self.wait()

        # Show how to reach point (5,2)
        thief_point2 = Dot(plane.coords_to_point(5, 2), color=RED)
        thief_label2 = Text("Thief (5,2)", font_size=25, color=RED)
        thief_label2.next_to(thief_point2, UP, buff=0.2)
        
        vector2_1_scaled = Arrow(plane.coords_to_point(0, 0), 
                            plane.coords_to_point(5, 0), 
                            buff=0, 
                            color=BLUE_A)
        vector2_2_scaled = Arrow(plane.coords_to_point(5, 0), 
                            plane.coords_to_point(5, 2), 
                            buff=0, 
                            color=RED_A)
        
        self.play(Create(thief_point2), Write(thief_label2))
        self.play(
            Transform(vector2_1, vector2_1_scaled),
            Transform(vector2_2, vector2_2_scaled)
        )
        
        # Show that we can reach any point
        span_text = Text("This set can reach ANY point in the plane!", 
                        font_size=25)
        span_text.to_edge(DOWN, buff=0.5)
        self.play(Write(span_text))
        self.wait(2)
        
        # Clear for Example 3
        self.play(
            *[FadeOut(mob) for mob in [example2_text, vector2_1, vector2_2,
                                    vector2_1_label, vector2_2_label,
                                    thief_point2, thief_label2, span_text]]
        )

        # Example 3: B3 = {(2,1)}
        example3_text = Text("Example 3: B₃ = {(2,1)}", font_size=30)
        example3_text.next_to(title, DOWN, buff=0.5)
        
        vector3 = Arrow(plane.coords_to_point(0, 0), 
                    plane.coords_to_point(2, 1), 
                    buff=0, 
                    color=BLUE)
        vector3_label = MathTex("(2,1)").next_to(vector3, UP, buff=0.2)
        
        self.play(Write(example3_text))
        self.play(Create(vector3), Write(vector3_label))
        self.wait()

        # Show thief at unreachable point (3,4)
        thief_point3 = Dot(plane.coords_to_point(3, 4), color=RED)
        thief_label3 = Text("Thief (3,4)", font_size=25, color=RED)
        thief_label3.next_to(thief_point3, UP, buff=0.2)
        
        self.play(Create(thief_point3), Write(thief_label3))
        
        # Show line of possible points
        line = Line(
            plane.coords_to_point(-10, -5),
            plane.coords_to_point(10, 5),
            color=YELLOW_A
        )
        unreachable = Text("Can only reach points on this line!", 
                        font_size=25)
        unreachable.to_edge(DOWN, buff=0.5)
        
        self.play(Create(line), Write(unreachable))
        self.wait(2)

        self.wait(2)

        # Clear for Example 4
        self.play(
            *[FadeOut(mob) for mob in [example3_text, vector3, vector3_label,
                                    thief_point3, thief_label3, line, unreachable]]
        )

        # Example 4: B4 = {(1,1), (2,2), (-1,-1)}
        example4_text = Text("Example 4: B₄ = {(1,1), (2,2), (-1,-1)}", font_size=30)
        example4_text.next_to(title, DOWN, buff=0.5)
        
        vector4_1 = Arrow(plane.coords_to_point(0, 0), 
                        plane.coords_to_point(1, 1), 
                        buff=0, 
                        color=BLUE)
        vector4_2 = Arrow(plane.coords_to_point(0, 0), 
                        plane.coords_to_point(2, 2), 
                        buff=0, 
                        color=RED)
        vector4_3 = Arrow(plane.coords_to_point(0, 0), 
                        plane.coords_to_point(-1, -1), 
                        buff=0, 
                        color=GREEN)
        
        vector4_1_label = MathTex("(1,1)").next_to(vector4_1, UP, buff=0.2)
        vector4_2_label = MathTex("(2,2)").next_to(vector4_2, UP, buff=0.2)
        vector4_3_label = MathTex("(-1,-1)").next_to(vector4_3, DOWN, buff=0.2)
        
        self.play(Write(example4_text))
        self.play(
            Create(vector4_1), Create(vector4_2), Create(vector4_3),
            Write(vector4_1_label), Write(vector4_2_label), Write(vector4_3_label)
        )
        
        # Show linear dependence
        dependent_text = Text("These vectors are linearly dependent!\nThey all lie on the same line.", 
                            font_size=25)
        dependent_text.to_edge(DOWN, buff=0.5)
        
        line_dependent = Line(
            plane.coords_to_point(-3, -3),
            plane.coords_to_point(3, 3),
            color=YELLOW_A
        )
        
        self.play(Create(line_dependent), Write(dependent_text))
        self.wait(2)

        # Clear for Example 5
        self.play(
            *[FadeOut(mob) for mob in [example4_text, vector4_1, vector4_2, vector4_3,
                                    vector4_1_label, vector4_2_label, vector4_3_label,
                                    line_dependent, dependent_text]]
        )

        # Example 5: B5 = {(1,0), (0,1), (1,1)}
        example5_text = Text("Example 5: B₅ = {(1,0), (0,1), (1,1)}", font_size=30)
        example5_text.next_to(title, DOWN, buff=0.5)
        
        vector5_1 = Arrow(plane.coords_to_point(0, 0), 
                        plane.coords_to_point(1, 0), 
                        buff=0, 
                        color=BLUE)
        vector5_2 = Arrow(plane.coords_to_point(0, 0), 
                        plane.coords_to_point(0, 1), 
                        buff=0, 
                        color=RED)
        vector5_3 = Arrow(plane.coords_to_point(0, 0), 
                        plane.coords_to_point(1, 1), 
                        buff=0, 
                        color=GREEN)
        
        vector5_1_label = MathTex("(1,0)").next_to(vector5_1, DOWN, buff=0.2)
        vector5_2_label = MathTex("(0,1)").next_to(vector5_2, LEFT, buff=0.2)
        vector5_3_label = MathTex("(1,1)").next_to(vector5_3, RIGHT, buff=0.2)
        
        self.play(Write(example5_text))
        self.play(
            Create(vector5_1), Create(vector5_2), Create(vector5_3),
            Write(vector5_1_label), Write(vector5_2_label), Write(vector5_3_label)
        )
        
        # Show that (1,1) is redundant
        redundant_text = Text("The vector (1,1) is redundant!\nIt can be obtained as (1,0) + (0,1)", 
                            font_size=25)
        redundant_text.to_edge(DOWN, buff=0.5)
        
        # Show the combination
        vector5_1_copy = Arrow(plane.coords_to_point(0, 0), 
                            plane.coords_to_point(1, 0), 
                            buff=0, 
                            color=YELLOW)
        vector5_2_copy = Arrow(plane.coords_to_point(1, 0), 
                            plane.coords_to_point(1, 1), 
                            buff=0, 
                            color=YELLOW)
        
        self.play(Write(redundant_text))
        self.play(
            Create(vector5_1_copy),
            Create(vector5_2_copy)
        )
        self.wait(2)

        self.wait(2)