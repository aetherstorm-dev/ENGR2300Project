from manim import *
import numpy as np
import math

class ShowDataAndSLP(Scene):

    def construct(self):

        self.wait(0.5)

        questionMark = Text("? = ", font_size = 120)
        questionMark.move_to([-2,0,0])
        questionMark.generate_target()

        genericData = Text("0 0 0 1 1 1 0 1")
        genericData.move_to([1.3,-0.15,0])

        # textGroup = VGroup(questionMark, genericData)

        self.play(Write(questionMark), run_time = 0.8)
        self.play(Write(genericData), run_time = 1.5)
        self.wait(1.5)

        questionMark.target.shift([-2,0,0])

        self.play(AnimationGroup(FadeOut(genericData), MoveToTarget(questionMark)))
        self.wait(0.5)

        twoImage = ImageMobject("assets\\images\\two.png")
        twoImage.scale(24).next_to(questionMark, RIGHT, buff=0.5).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])

        imageHider = Square().set_color(BLACK).set_fill(BLACK, opacity=1).move_to(twoImage).match_height(twoImage).match_width(twoImage)

        self.add(twoImage)
        self.add(imageHider)
        self.play(FadeOut(imageHider), run_time = 0.8)

        twoImageMatrix = Matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                v_buff = 0.6, h_buff = 0.6).scale(0.56).next_to(twoImage, RIGHT, buff=0.6)
        
        self.wait(0.3)

        self.play(Create(twoImageMatrix), run_time=1.2)
        self.wait(2.5)

        self.play(FadeIn(imageHider), Uncreate(twoImageMatrix), run_time = 1)
        self.remove(twoImage, imageHider)

        vertices = [1, 2, 3, 4, 5, 6, 7]
        edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
        lt = {1: [0, 1.5, 0], 2: [1, 3, 0], 3: [2, 2.4, 0], 4: [3, 0, 0], 5: [4, 0.6, 0], 6: [5, 1.8, 0], 7: [6, 2.4, 0]}

        audioGraph = Graph(vertices, edges, layout = lt, vertex_config = {"fill_color": BLUE}, edge_config = {"stroke_color": BLUE}).move_to([0, 0, 0])
        audioPlane = NumberPlane(x_range = (0, 6, 1), y_range = (0, 1, 0.5), x_length = 6, y_length = 3, background_line_style = {
                "stroke_color": GREY}).move_to(audioGraph)

        audioVector = Matrix([[0.5], [1.0], [0.8], [0.0], [0.2], [0.6], [0.8]], v_buff=0.9).scale(0.8).next_to(audioPlane, RIGHT, buff=0.75)

        questionMark.target.shift([-0.5,0,0])

        self.play(MoveToTarget(questionMark), run_time = 0.5)
        self.play(Create(audioGraph), Create(audioPlane), run_time = 1)
        self.wait(0.5)

        self.play(Create(audioVector), run_time=0.8)
        self.wait(2.0)

        self.play(Uncreate(audioGraph), Uncreate(audioPlane), Uncreate(audioVector), run_time = 0.9, lag_ratio = 0.3)
        self.wait(0.2)

        sampleText = Text("The quick brown fox jumps\nover the lazy dog.", font_size = 39, weight = NORMAL, line_spacing = 1.2,
                t2c={'[0:3]': BLUE, '[4:9]': YELLOW, '[10:15]': DARK_BLUE, '[16:19]': GREEN, '[20:25]': ORANGE,
                '[26:30]': TEAL, '[31:34]': GRAY_B, '[35:39]': PURPLE, '[40:43]': DARK_BLUE, '[43:44]': GREEN_C})
        sampleText.move_to([0.4, 1.1, 0])

        sampleTokens = Text("976 4853 19705 68347 65613\n1072 290 29082 6446 13", font_size = 32, weight = NORMAL, line_spacing = 1.1,
        t2c={'[0:3]': BLUE, '[4:8]': YELLOW, '[9:14]': DARK_BLUE, '[15:20]': GREEN, '[21:26]': ORANGE,
                '[27:31]': TEAL, '[32:35]': GRAY_B, '[36:41]': PURPLE, '[42:46]': DARK_BLUE, '[47:49]': GREEN_C})
        sampleTokens.move_to([0, -1.1, 0]).align_to(sampleText, LEFT)

        self.play(Write(sampleText), run_time = 1)
        self.play(Write(sampleTokens), run_time = 1)
        self.wait(1.5)

        self.play(Unwrite(sampleText), Uncreate(sampleTokens), run_time = 0.6, lag_ratio = 0.2)

        questionMark.target.shift([0.9,0,0])

        aEquals = Text("A = ", font_size = 120).move_to(questionMark)

        exampleVector = Matrix([[0], [1], [1], [0], [0], [0], [1], [1], [1], [0], [1], [1], [0], [0], [1], [1]],
                v_buff=0.7).scale(0.63).next_to(questionMark,buff=0.7)

        self.play(MoveToTarget(questionMark), Transform(questionMark, aEquals), run_time = 0.8, lag_ratio = 0.2)
        self.play(Create(exampleVector), run_time = 1.5)
        self.wait(5)

        edges = []
        partitions = []
        c = 0
        layers = [2, 1]

        for i in layers:
            partitions.append(list(range(c + 1, c + i + 1)))
            c += i
        for i, v in enumerate(layers[1:]):
                last = sum(layers[:i+1])
                for j in range(v):
                    for k in range(last - layers[i], last):
                        edges.append((k + 1, j + last + 1))

        vertices = np.arange(1, sum(layers) + 1)

        basicSLP = Graph(vertices, edges, layout='partite', partitions=partitions, layout_scale=1.75,
                vertex_config = {'radius': 0.27, "fill_color": WHITE}, edge_config = {"stroke_color": GREY}).move_to([1.5, 0, 0])

        self.play(Create(basicSLP), runTime = 2)
        self.wait(2)

        # First and second values are x and y coordinates; third value is true/false (data)
        pointVectorTrain = [[0, 0, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0], [0.5, 0.5, 1]]
        pointVectorTest = [[0.2, 0.25], [0.8, 0.75], [0.4, 0.3], [0.8, 0.5], [0.9, 0.9],
                [0.1, 0.5], [0.3, 0.1], [0.05, 0.2], [0.9, 0.6], [0.65, 0.7],
                [0.7, 0.1], [0.2, 0.8], [0.4, 0.4], [0.4, 0.9], [0.7, 0.3],
                [0.5, 0.8], [0.6, 0.2]]
        
        xWeightTracker = ValueTracker(0.6)
        yWeightTracker = ValueTracker(1.2)

        pointXCoords = []
        pointYCoords = []

        for i in range(0, len(pointVectorTest)):
            pointXCoords.append(pointVectorTest[i][0])
            pointYCoords.append(pointVectorTest[i][1])
        
        pointVectorMatrix = Matrix(pointVectorTest, v_buff=0.8, h_buff=1.25).scale(0.63).move_to(exampleVector)
        pointVectorMatrix.generate_target()
        pointVectorMatrix.target.shift([-2.5,0,0])

        self.play(FadeOut(questionMark), FadeOut(aEquals), run_time = 0.8, lag_ratio = 0.2)

        basicSLP.generate_target()
        basicSLP.target.shift([-2, 0, 0])

        self.play(Uncreate(exampleVector), run_time = 0.6)
        self.play(Create(pointVectorMatrix), MoveToTarget(pointVectorMatrix), MoveToTarget(basicSLP), run_time = 1.2)

        linearClassificationAxes = Axes(x_range = (0, 1, 0.1), y_range = (0, 1, 0.1), x_length = 3.6,
               tips=False, y_length = 3.6).move_to([5, 0, 0])
    
        pointsOnAxis = linearClassificationAxes.plot_line_graph(pointXCoords, pointYCoords, vertex_dot_radius=0.05)

        self.wait(0.8)

        self.play(Create(linearClassificationAxes), run_time = 1)
        self.play(Create(pointsOnAxis["vertex_dots"]), run_time = 0.8)

        self.wait(1)

        pointVectorMatrixConsumable = pointVectorMatrix.copy()
        consumableEntries = pointVectorMatrixConsumable.get_entries()
        self.add(pointVectorMatrixConsumable)

        for i in range(0, len(pointVectorTest)):
            speedMult = 0.33
            speedMult2 = 0.5
            if i < 3:
                speedMult = 1.66
                speedMult2 = 1.66
            
            xVertexColor = "#" + f'{str(hex(int(pointXCoords[i]*255)))[2:]:0>2}'*3
            yVertexColor = "#" + f'{str(hex(int(pointYCoords[i]*255)))[2:]:0>2}'*3

            consumableEntries[2*i].generate_target()
            consumableEntries[2*i + 1].generate_target()
            consumableEntries[2*i].target.move_to(basicSLP.vertices[2]).scale(0.5)
            consumableEntries[2*i + 1].target.move_to(basicSLP.vertices[1]).scale(0.5)

            selectedPoint = pointsOnAxis["vertex_dots"][i]

            pointBox = Square(stroke_width=1.6).match_width(selectedPoint).match_height(selectedPoint).scale(1.6).move_to(selectedPoint)

            if i < 3:
                self.play(FadeIn(pointBox), runTime = 0.1*speedMult)

            self.play(basicSLP.vertices[2].animate.set_color(xVertexColor), MoveToTarget(consumableEntries[2*i]), run_time = 0.4*speedMult)
            self.play(basicSLP.vertices[1].animate.set_color(yVertexColor), MoveToTarget(consumableEntries[2*i + 1]), run_time = 0.4*speedMult)
            self.wait(0.2*speedMult)

            resultColor = ""
            result = -1

            if (pointXCoords[i] * xWeightTracker.get_value()) + (pointYCoords[i] * yWeightTracker.get_value()) >= 1:
                resultColor = ORANGE
                result = 1
            else: 
                resultColor = BLUE
                result = 0

            self.play(FadeOut(consumableEntries[2*i], consumableEntries[2*i + 1]), basicSLP.vertices[3].animate.set_color(resultColor), run_time = 0.4*speedMult)
            self.wait(0.1*speedMult)
            self.remove(consumableEntries[2*i], consumableEntries[2*i + 1])

            resultText = Text(str(result), font_size = 30).move_to(basicSLP.vertices[3])
            resultText.generate_target()
            resultText.target.shift([1.2, 0, 0]).scale(2.0)

            self.play(FadeIn(resultText), MoveToTarget(resultText), run_time = 0.8*(speedMult/2))
            self.wait(0.1*speedMult)

            if i < 3:
                self.play(selectedPoint.animate.set_color(resultColor), FadeOut(pointBox), run_time = 0.3*speedMult2)
            else:
                self.play(selectedPoint.animate.set_color(resultColor), run_time = 0.3*speedMult2)
            self.wait(0.1*speedMult)

            self.play(FadeOut(resultText), run_time = 0.4*(speedMult2))
        
        self.wait(2)

        sumTex = MathTex("a_{0}^{(0)}" + "+" + "a_{1}^{(0)}" + "=" + "a_{0}^{(1)}",
                substrings_to_isolate=["a_{0}^{(0)}", "a_{1}^{(0)}", "a_{0}^{(1)}"]).move_to([0, 3.5, 0])

        self.play(basicSLP.vertices[3].animate.set_color(WHITE), Write(sumTex), run_time = 0.8)
        self.wait(0.8)

        pointVectorMatrixConsumable = pointVectorMatrix.copy()
        consumableEntries = pointVectorMatrixConsumable.get_entries()
        self.add(pointVectorMatrixConsumable)

        xVertexColor = "#" + f'{str(hex(int(pointXCoords[0]*255)))[2:]:0>2}'*3
        yVertexColor = "#" + f'{str(hex(int(pointYCoords[0]*255)))[2:]:0>2}'*3

        consumableEntries[0].generate_target()
        consumableEntries[1].generate_target()
        consumableEntries[0].target.move_to(basicSLP.vertices[2]).scale(0.5)
        consumableEntries[1].target.move_to(basicSLP.vertices[1]).scale(0.5)

        sumFirstElement = sumTex.get_part_by_tex("a_{0}^{(0)}")
        firstElementValue = Tex(str(pointXCoords[0])).move_to(sumFirstElement)
        sumSecondElement = sumTex.get_part_by_tex("a_{1}^{(0)}")
        secondElementValue = Tex(str(pointYCoords[0])).move_to(sumSecondElement)

        self.play(basicSLP.vertices[2].animate.set_color(xVertexColor), MoveToTarget(consumableEntries[0]),
                Transform(sumFirstElement, firstElementValue), run_time = 0.8)
        self.play(basicSLP.vertices[1].animate.set_color(yVertexColor), MoveToTarget(consumableEntries[1]),
                Transform(sumSecondElement, secondElementValue), run_time = 0.8)
        self.wait(0.4)

        self.play(FadeOut(consumableEntries[0], consumableEntries[1]), basicSLP.vertices[3].animate.set_color(GREY), run_time = 0.8)
        self.wait(0.2)
        self.remove(consumableEntries[0], consumableEntries[1])

        sumResultElement = sumTex.get_part_by_tex("a_{0}^{(1)}")
        resultElementValue = Tex(str(pointXCoords[0] + pointYCoords[0]) + "?").move_to(sumResultElement).shift([0.05, 0, 0])

        resultText = Text(str(pointXCoords[0] + pointYCoords[0]) + "?", font_size = 20).move_to(basicSLP.vertices[3])
        resultText.generate_target()
        resultText.target.shift([1.2, 0, 0]).scale(2.0)

        self.play(FadeIn(resultText), MoveToTarget(resultText), Transform(sumResultElement, resultElementValue), run_time = 0.8)
        self.wait(4)

        sumTexWeights = MathTex("w_{0}a_{0}^{(0)} + w_{1}a_{1}^{(0)} = a_{0}^{(1)}").move_to(sumTex)

        self.play(FadeOut(resultText), Transform(sumTex, sumTexWeights), run_time = 0.8)
        self.wait(0.5)

        firstWeight = MathTex("w_{0}", font_size = 30).move_to([-0.3, 1.2, 0])
        secondWeight = MathTex("w_{1}", font_size = 30).move_to([-0.3, -1.2, 0])

        self.play(Write(firstWeight), Write(secondWeight), run_time = 0.6)
        self.wait(1)

        weightVector = [0.6, 1.2]
        sumTexWeightsValued = MathTex("0.6a_{0}^{(0)} + 1.2a_{1}^{(0)} = a_{0}^{(1)}").move_to(sumTex)

        firstWeightValued = MathTex(str(weightVector[0]), font_size = 30).move_to(firstWeight)
        secondWeightValued = MathTex(str(weightVector[1]), font_size = 30).move_to(secondWeight)

        SLPxEdge = basicSLP.edges[1,3]
        SLPyEdge = basicSLP.edges[2,3]

        self.play(Transform(sumTex, sumTexWeightsValued), ReplacementTransform(firstWeight, firstWeightValued),
                ReplacementTransform(secondWeight, secondWeightValued), SLPxEdge.animate.set_color(ORANGE),
                SLPyEdge.animate.set_color(BLUE), run_time = 0.8)
        
        self.wait(6)

        sumTexStep = MathTex("u_{1}(0.6a_{0}^{(0)} + 1.2a_{1}^{(0)} )= a_{0}^{(1)}").move_to(sumTex)
        stepFunctionTex = MathTex("u_{1}(x)= \left\{ \\begin{array}{cl} 0 & : \ x < 0 \\\\ 1 & : \ x \geq 0 \end{array} \\right.").move_to([0, -3.5, 0])

        self.play(Transform(sumTex, sumTexStep), run_time = 0.8)
        self.wait(0.5)

        self.play(Write(stepFunctionTex), run_time = 0.8)
        self.wait(1.5)

        xVertexColor = "#" + f'{str(hex(int(pointXCoords[1]*255)))[2:]:0>2}'*3
        yVertexColor = "#" + f'{str(hex(int(pointYCoords[1]*255)))[2:]:0>2}'*3

        consumableEntries[2].generate_target()
        consumableEntries[3].generate_target()
        consumableEntries[2].target.move_to(basicSLP.vertices[2]).scale(0.5)
        consumableEntries[3].target.move_to(basicSLP.vertices[1]).scale(0.5)

        self.play(basicSLP.vertices[2].animate.set_color(xVertexColor), MoveToTarget(consumableEntries[2]), run_time = 0.8)
        self.play(basicSLP.vertices[1].animate.set_color(yVertexColor), MoveToTarget(consumableEntries[3]), run_time = 0.8)

        sumTexStepValues = MathTex("u_{1}(" + str(round(xWeightTracker.get_value()*pointXCoords[1], 2)) + " + " + str(round(yWeightTracker.get_value()*pointYCoords[1], 2)) + ")= a_{0}^{(1)}").move_to(sumTex)

        self.play(Transform(sumTex, sumTexStepValues), run_time = 0.8)
        self.wait(0.4)

        resultColor = ""
        result = -1

        if (pointXCoords[1] * xWeightTracker.get_value()) + (pointYCoords[1] * yWeightTracker.get_value()) >= 1:
            resultColor = ORANGE
            result = 1
        else: 
            resultColor = BLUE
            result = 0

        self.play(FadeOut(consumableEntries[2], consumableEntries[3]), basicSLP.vertices[3].animate.set_color(resultColor), run_time = 0.8)
        self.wait(0.2)
        self.remove(consumableEntries[2], consumableEntries[3])

        resultText = Text(str(result), font_size = 30).move_to(basicSLP.vertices[3])
        resultText.generate_target()
        resultText.target.shift([1.2, 0, 0]).scale(2.0)

        self.play(FadeIn(resultText), MoveToTarget(resultText), run_time = 0.8)
        self.wait(0.2)

        sumTexStepValuesResult = MathTex("u_{1}(" + str(round(0.6*pointXCoords[1], 2)) + " + " + str(round(1.2*pointYCoords[1], 2)) + ")=" + str(result)).move_to(sumTex)

        self.play(Transform(sumTex, sumTexStepValuesResult), run_time = 0.8)
        self.wait(4)

        linearClassifierBoundary = linearClassificationAxes.plot(lambda x: (1 - (xWeightTracker.get_value() * x))/yWeightTracker.get_value(), x_range=[0, 1], use_smoothing=False)
        
        self.play(FadeOut(sumTex), FadeOut(stepFunctionTex), basicSLP.vertices[3].animate.set_color(WHITE), FadeOut(resultText), Create(linearClassifierBoundary), run_time = 0.6)
        self.wait(5)

        xWeightTracker.set_value(1.2)
        yWeightTracker.set_value(0.6)

        firstWeightValued1 = MathTex(str(1.2), font_size = 30).move_to(firstWeightValued)
        secondWeightValued1 = MathTex(str(0.6), font_size = 30).move_to(secondWeightValued)

        linearClassifierBoundary1 = linearClassificationAxes.plot(lambda x: (1 - (xWeightTracker.get_value() * x))/yWeightTracker.get_value(), x_range=[1/3, 5/6], use_smoothing=False)

        pointColorChangesList = []
        for i in range(0, len(pointVectorTest)):
            selectedPoint = pointsOnAxis["vertex_dots"][i]

            resultColor = ""

            if (pointXCoords[i] * xWeightTracker.get_value()) + (pointYCoords[i] * yWeightTracker.get_value()) >= 1:
                resultColor = ORANGE
            else: 
                resultColor = BLUE
            
            pointColorChangesList.append(selectedPoint.animate.set_color(resultColor))

        pointColorChanges = AnimationGroup(pointColorChangesList)
        self.play(SLPxEdge.animate.set_color(BLUE), SLPyEdge.animate.set_color(ORANGE), ReplacementTransform(linearClassifierBoundary, linearClassifierBoundary1), 
                ReplacementTransform(firstWeightValued, firstWeightValued1), ReplacementTransform(secondWeightValued, secondWeightValued1), pointColorChanges, run_time = 1)
        self.wait(1)

        xWeightTracker.set_value(2.4)
        yWeightTracker.set_value(2.4)

        firstWeightValued2 = MathTex(str(2.4), font_size = 30).move_to(firstWeightValued1)
        secondWeightValued2 = MathTex(str(2.4), font_size = 30).move_to(secondWeightValued1)
        
        linearClassifierBoundary2 = linearClassificationAxes.plot(lambda x: (1 - (xWeightTracker.get_value() * x))/yWeightTracker.get_value(), x_range=[0, 5/12], use_smoothing=False)

        pointColorChangesList = []
        for i in range(0, len(pointVectorTest)):
            selectedPoint = pointsOnAxis["vertex_dots"][i]

            resultColor = ""

            if (pointXCoords[i] * xWeightTracker.get_value()) + (pointYCoords[i] * yWeightTracker.get_value()) >= 1:
                resultColor = ORANGE
            else: 
                resultColor = BLUE
            
            pointColorChangesList.append(selectedPoint.animate.set_color(resultColor))

        pointColorChanges = AnimationGroup(pointColorChangesList)
        self.play(SLPxEdge.animate.set_color(ORANGE), SLPyEdge.animate.set_color(ORANGE), ReplacementTransform(linearClassifierBoundary1, linearClassifierBoundary2),
                ReplacementTransform(firstWeightValued1, firstWeightValued2), ReplacementTransform(secondWeightValued1, secondWeightValued2), pointColorChanges, run_time = 1)
        self.wait(1)

        xWeightTracker.set_value(0.6)
        yWeightTracker.set_value(1.2)

        firstWeightValued3 = MathTex(str(0.6), font_size = 30).move_to(firstWeightValued2)
        secondWeightValued3 = MathTex(str(1.2), font_size = 30).move_to(secondWeightValued2)
        
        linearClassifierBoundary3 = linearClassificationAxes.plot(lambda x: (1 - (xWeightTracker.get_value() * x))/yWeightTracker.get_value(), x_range=[0, 1], use_smoothing=False)

        self.play(SLPxEdge.animate.set_color(BLUE), SLPyEdge.animate.set_color(ORANGE), ReplacementTransform(linearClassifierBoundary2, linearClassifierBoundary3),
                ReplacementTransform(firstWeightValued2, firstWeightValued3), ReplacementTransform(secondWeightValued2, secondWeightValued3), FadeOut(pointsOnAxis["vertex_dots"]), run_time = 1.2)

        XORVectorTest = [[0.2, 0.2], [0.25, 0.4], [0.35, 0.2], [0.4, 0.3],
                [0.2, 0.8], [0.1, 0.75], [0.3, 0.65], [0.25, 0.7],
                [0.8, 0.2], [0.7, 0.15], [0.9, 0.15], [0.75, 0.25],
                [0.8, 0.8], [0.65, 0.7], [0.9, 0.65], [0.75, 0.6]]

        pointXCoords = []
        pointYCoords = []

        for i in range(0, len(XORVectorTest)):
            pointXCoords.append(XORVectorTest[i][0])
            pointYCoords.append(XORVectorTest[i][1])

        pointsOnAxis = linearClassificationAxes.plot_line_graph(pointXCoords, pointYCoords, vertex_dot_radius=0.05)

        self.play(Create(pointsOnAxis["vertex_dots"]), run_time = 0.8)
        self.wait(1)

        self.play(FadeOut(linearClassifierBoundary3), run_time = 0.8)
        self.wait(4)

       
        
class ShowMLP(Scene):

    def construct(self):

        XORVectorTest = [[0.2, 0.2], [0.25, 0.4], [0.35, 0.2], [0.4, 0.3],
                [0.2, 0.8], [0.1, 0.75], [0.3, 0.65], [0.25, 0.7],
                [0.8, 0.2], [0.7, 0.15], [0.9, 0.15], [0.75, 0.25],
                [0.8, 0.8], [0.65, 0.7], [0.9, 0.65], [0.75, 0.6]]

        pointXCoords = []
        pointYCoords = []

        for i in range(0, len(XORVectorTest)):
            pointXCoords.append(XORVectorTest[i][0])
            pointYCoords.append(XORVectorTest[i][1])

        edges = []
        partitions = []
        c = 0
        layers = [2, 4, 1]

        for i in layers:
            partitions.append(list(range(c + 1, c + i + 1)))
            c += i
        for i, v in enumerate(layers[1:]):
                last = sum(layers[:i+1])
                for j in range(v):
                    for k in range(last - layers[i], last):
                        edges.append((k + 1, j + last + 1))

        vertices = np.arange(1, sum(layers) + 1)

        basicMLP = Graph(vertices, edges, layout = 'partite', partitions = partitions, layout_scale = 2,
                vertex_config = {'radius': 0.25, "fill_color": WHITE}, edge_config = {"stroke_color": GREY}).move_to([0, 0, 0])

        self.play(Create(basicMLP), runTime = 3.6)
        self.wait(1)

        highlightLine1 = Line(start = [-0.7, -10, 0], end = [-0.7, 10, 0], stroke_color = YELLOW)
        highlightLine2 = Line(start = [0.7, -10, 0], end = [0.7, 10, 0], stroke_color = YELLOW)

        self.play(FadeIn(highlightLine1), FadeIn(highlightLine2), run_time = 0.8)
        self.wait(1.5)

        self.play(FadeOut(highlightLine1), FadeOut(highlightLine2), run_time = 0.8)
        self.wait(2)

        basicMLP.generate_target()
        basicMLP.target.shift([-4, 0, 0])

        self.play(MoveToTarget(basicMLP), run_time = 1)
        self.wait(2)

        outputFeatureAxes = Axes(x_range = (0, 1, 0.1), y_range = (0, 1, 0.1), tips=False, x_length = 2,
                y_length = 2).move_to([5, 0, 0])
        
        self.play(Create(outputFeatureAxes), run_time = 1)

        outputFeatureBoundary = outputFeatureAxes.plot(lambda x: (1 - (0.6 * x))/1.2, x_range=[0, 1], use_smoothing=False)
        
        self.play(Create(outputFeatureBoundary), run_time = 0.6)
        self.wait(3)

        firstNeuronFeatureAxes = Axes(x_range = (0, 1, 0.1), y_range = (0, 1, 0.1), tips=False, x_length = 1.5,
                y_length = 1.5).move_to([2.5, 3, 0])
        secondNeuronFeatureAxes = Axes(x_range = (0, 1, 0.1), y_range = (0, 1, 0.1), tips=False, x_length = 1.5,
                y_length = 1.5).move_to([2.5, 1, 0])
        thirdNeuronFeatureAxes = Axes(x_range = (0, 1, 0.1), y_range = (0, 1, 0.1), tips=False, x_length = 1.5,
                y_length = 1.5).move_to([2.5, -1, 0])
        fourthNeuronFeatureAxes = Axes(x_range = (0, 1, 0.1), y_range = (0, 1, 0.1), tips=False, x_length = 1.5,
                y_length = 1.5).move_to([2.5, -3, 0])
        
        firstNeuronBoundary = firstNeuronFeatureAxes.plot(lambda x: ((1 - (1 * x))/-1) + 1.5, x_range=[0, 0.5], use_smoothing=False)
        secondNeuronBoundary = secondNeuronFeatureAxes.plot(lambda x: ((1 - (-1 * x))/1) - 1.5, x_range=[0.5, 1], use_smoothing=False)
        thirdNeuronBoundary = thirdNeuronFeatureAxes.plot(lambda x: ((1 - (-1 * x))/-1) + 1.5, x_range=[0, 0.5], use_smoothing=False)
        fourthNeuronBoundary = fourthNeuronFeatureAxes.plot(lambda x: ((1 - (1 * x))/1) + 0.5, x_range=[0.5, 1], use_smoothing=False)

        self.play(Create(firstNeuronFeatureAxes), run_time = 0.8)
        self.play(Create(secondNeuronFeatureAxes), Create(firstNeuronBoundary), run_time = 0.8)
        self.play(Create(thirdNeuronFeatureAxes), Create(secondNeuronBoundary), run_time = 0.8)
        self.play(Create(fourthNeuronFeatureAxes), Create(thirdNeuronBoundary), run_time = 0.8)
        self.play(Create(fourthNeuronBoundary), run_time = 0.8)
        self.wait(2.5)

        self.play(FadeOut(outputFeatureAxes), FadeOut(outputFeatureBoundary), FadeOut(firstNeuronFeatureAxes), FadeOut(firstNeuronBoundary),
                FadeOut(secondNeuronFeatureAxes), FadeOut(secondNeuronBoundary), FadeOut(thirdNeuronFeatureAxes), FadeOut(thirdNeuronBoundary),
                FadeOut(fourthNeuronFeatureAxes), FadeOut(fourthNeuronBoundary), run_time = 1.2)
        self.wait(3)

        sigmoidFunctionTex = MathTex("\\sigma(x)=\\frac{1}{1+e^{-x}}").move_to([0, 2.5, 0])
        sigmoidFunctionAxes = Axes(x_range = (-6, 6, 1), y_range = (0, 2, 1), x_length = 10, y_length = 3,
                tips=False, axis_config={"include_numbers": True}).move_to([0, -1.8, 0])
        sigmoidFunctionPlot = sigmoidFunctionAxes.plot(lambda x: 1 / (1 + math.exp(-x)), x_range=[-6, 6], use_smoothing = True)

        basicMLP.target.shift([-1.5, 0, 0])

        self.play(MoveToTarget(basicMLP), Write(sigmoidFunctionTex), Create(sigmoidFunctionAxes), run_time = 1.2)
        self.play(Create(sigmoidFunctionPlot), run_time = 1.2)
        self.wait(2)
        
        outputActivationTex = MathTex("\sigma(w_{0,0}a_{0}^{(0)} + w_{0,1}a_{1}^{(0)}) = a_{0}^{(1)}").shift([0, 1, 0])

        self.play(Write(outputActivationTex), run_time = 0.8)
        self.wait(2)

        sigmoidDot = Dot().set_color(WHITE).move_to(sigmoidFunctionPlot.get_left()).shift([0, -0.75, 0])

        self.play(FadeIn(sigmoidDot), run_time = 0.75)
        self.play(MoveAlongPath(sigmoidDot, sigmoidFunctionPlot), run_time = 4, rate_func = rate_functions.ease_in_out_cubic)
        self.play(FadeOut(sigmoidDot), run_time = 0.75)
        self.wait(2)

        # Each inner array is a column
        weightMatrix0 = [[-15.682, -13.981], [-11.246, -7.9654], [11.513, -8.3954], [10.721, -11.251]]
        weightMatrix1 = [[-15.589], [14.603], [-17.061], [14.605]]

        biasArray = [11.565, 11.734, 1.6322, -2.6202]

        basicMLP.target.shift([1, 0, 0])

        outputActivationTex.generate_target()
        outputActivationTex.target.shift([0, 1.5, 0])

        self.play(MoveToTarget(basicMLP), Unwrite(sigmoidFunctionTex), MoveToTarget(outputActivationTex), FadeOut(sigmoidDot), Uncreate(sigmoidFunctionAxes), Uncreate(sigmoidFunctionPlot), run_time = 1.2)
        self.wait(0.5)

        outputBiasTex = MathTex("\sigma(w_{0,0}a_{0}^{(0)} + w_{0,1}a_{1}^{(0)} + b_{0}) = a_{0}^{(1)}").move_to(outputActivationTex)

        self.play(ReplacementTransform(outputActivationTex, outputBiasTex), run_time = 0.8)
        self.wait(1)

        biasIndicatorOne = Square(0.2).set_fill(ORANGE, opacity = 0.9).next_to(basicMLP.vertices[6], direction = LEFT, buff = 0.16)
        biasIndicatorTwo = Square(0.2).set_fill(ORANGE, opacity = 0.9).next_to(basicMLP.vertices[5], direction = LEFT, buff = 0.16)
        biasIndicatorThree = Square(0.2).set_fill(ORANGE, opacity = 0.4).next_to(basicMLP.vertices[4], direction = LEFT, buff = 0.16)
        biasIndicatorFour = Square(0.2).set_fill(BLUE, opacity = 0.5).next_to(basicMLP.vertices[3], direction = LEFT, buff = 0.16)

        self.play(Create(biasIndicatorOne), run_time = 0.6)
        self.play(Create(biasIndicatorTwo), run_time = 0.6)
        self.play(Create(biasIndicatorThree), run_time = 0.6)
        self.play(Create(biasIndicatorFour), run_time = 0.6)
        self.wait(2.5)

        MLPedgeAnimations1 = []
        MLPedgeAnimations2 = []

        for i in range(1, 3):
            for j in range(3, 7):
                MLPedgeAnimations1.append(basicMLP.edges[i, j].animate.set_color(YELLOW))

        for i in range(3, 7):
            MLPedgeAnimations2.append(basicMLP.edges[i, 7].animate.set_color(YELLOW))

        self.play(MLPedgeAnimations1, run_time = 0.8)
        self.wait(0.4)

        self.play(MLPedgeAnimations2, run_time = 0.8)
        self.wait(1.6)

        MLPedgeAnimations1 = []
        MLPedgeAnimations2 = []

        edgeColor = WHITE
        for i in range(1, 3):
            for j in range(3, 7):
                if weightMatrix0[j-3][i-1] < 0.5:
                    edgeColor = BLUE
                elif weightMatrix0[j-3][i-1] > 0.5:
                    edgeColor = ORANGE
                MLPedgeAnimations1.append(basicMLP.edges[i, j].animate.set_color(edgeColor))

        for i in range(3, 7):
            if weightMatrix1[i-3][0] < 0.5:
                edgeColor = BLUE
            elif weightMatrix1[i-3][0] > 0.5:
                edgeColor = ORANGE
            MLPedgeAnimations2.append(basicMLP.edges[i, 7].animate.set_color(edgeColor))

        self.play(MLPedgeAnimations1, run_time = 0.8)
        self.wait(0.4)

        self.play(MLPedgeAnimations2, run_time = 0.8)
        self.wait(8)

        layerMatrixTex = MathTex("\\sigma(\\textbf{W} \\textbf{a}^{(0)} + \\textbf{b}) = \\textbf{a}^{(1)}").move_to(outputBiasTex)

        self.play(ReplacementTransform(outputBiasTex, layerMatrixTex), run_time = 0.8)
        self.wait(1)

        layerMatrixFullTex = MathTex("""\\large\\sigma\\normalsize\\begin{pmatrix}
                \\begin{bmatrix}
                w_{0,0} & w_{0,1} & \\cdots & w_{0,n}\\\\
                w_{1,0} & w_{1,1} & \\cdots & w_{1,n}\\\\
                \\vdots & \\vdots & \\ddots & \\vdots\\\\
                w_{k,0} & w_{k,0} & \\cdots & w_{k,n}
                \\end{bmatrix}
                \\begin{bmatrix}
                a_{0}^{(0)}\\\\
                a_{1}^{(0)}\\\\
                \\vdots\\\\
                a_{n}^{(0)}
                \\end{bmatrix}
                \\large +
                \\normalsize
                \\begin{bmatrix}
                b_{0}\\\\
                b_{1}\\\\
                \\vdots\\\\
                b_{n}
                \\end{bmatrix}
                \\end{pmatrix}
                \\large =
                \\normalsize
                a_{1}^{(0)}""").scale(0.7).move_to([0.5,-2,0])

        self.play(Write(layerMatrixFullTex), run_time = 1.6)
        self.wait(8)

        layerMatrixTex.generate_target()
        layerMatrixTex.target.shift([0, 1, 0])

        basicMLP.generate_target()
        basicMLP.target.shift([4, 0, 0])

        self.play(MoveToTarget(layerMatrixTex), Uncreate(biasIndicatorOne), Uncreate(biasIndicatorTwo), Uncreate(biasIndicatorThree),
                Uncreate(biasIndicatorFour), Unwrite(layerMatrixFullTex), run_time = 1.2)
        self.play(MoveToTarget(basicMLP), run_time = 1.2)
        self.wait(1)

        XORclassificationAxes = Axes(x_range = (0, 1, 0.1), y_range = (0, 1, 0.1), x_length = 3.6,
               tips=False, y_length = 3.6).move_to([5, 0, 0])
    
        pointsOnAxis = XORclassificationAxes.plot_line_graph(pointXCoords, pointYCoords, vertex_dot_radius=0.05)

        XORVectorMatrix = Matrix(XORVectorTest, v_buff=0.8, h_buff=1.25).scale(0.6).move_to([-5, 0, 0])

        self.play(Create(XORVectorMatrix), Create(XORclassificationAxes), run_time = 1)
        self.play(Create(pointsOnAxis["vertex_dots"]), run_time = 0.8)
        self.wait(1)

        XORVectorMatrixConsumable = XORVectorMatrix.copy()
        consumableEntries = XORVectorMatrixConsumable.get_entries()
        self.add(XORVectorMatrixConsumable)

        blueOrangeGradient = color_gradient([BLUE, GREY, ORANGE], 51)
        for i in range(0, len(XORVectorTest)):
            speedMult = 0.33
            speedMult2 = 0.5
            if i < 2:
                speedMult = 1.66
                speedMult2 = 1.66
            
            xVertexColor = "#" + f'{str(hex(int(pointXCoords[i]*255)))[2:]:0>2}'*3
            yVertexColor = "#" + f'{str(hex(int(pointYCoords[i]*255)))[2:]:0>2}'*3

            consumableEntries[2*i].generate_target()
            consumableEntries[2*i + 1].generate_target()
            consumableEntries[2*i].target.move_to(basicMLP.vertices[2]).scale(0.5)
            consumableEntries[2*i + 1].target.move_to(basicMLP.vertices[1]).scale(0.5)

            selectedPoint = pointsOnAxis["vertex_dots"][i]

            pointBox = Square(stroke_width=1.6).match_width(selectedPoint).match_height(selectedPoint).scale(1.6).move_to(selectedPoint)

            if i < 3:
                self.play(FadeIn(pointBox), runTime = 0.1*speedMult)

            self.play(basicMLP.vertices[2].animate.set_color(xVertexColor), MoveToTarget(consumableEntries[2*i]), run_time = 0.4*speedMult)
            self.play(basicMLP.vertices[1].animate.set_color(yVertexColor), MoveToTarget(consumableEntries[2*i + 1]), run_time = 0.4*speedMult)
            self.wait(0.2*speedMult)

            hiddenArray = []
            hiddenLayerColorAnimations = []
            for j in range(3, 7):
                result = (pointXCoords[i] * weightMatrix0[j-3][0]) + (pointYCoords[i] * weightMatrix0[j-3][1]) + biasArray[j-3]
                resultSigmoid = 1 / (1 + (math.e ** (-result)))
                hiddenArray.append(resultSigmoid)
                resultColor = blueOrangeGradient[math.floor(resultSigmoid * len(blueOrangeGradient))]

                hiddenLayerColorAnimations.append(basicMLP.vertices[j].animate.set_color(resultColor))

            self.play(FadeOut(consumableEntries[2*i], consumableEntries[2*i + 1]), hiddenLayerColorAnimations, run_time = 0.4*speedMult)
            self.wait(0.25*speedMult)
            self.remove(consumableEntries[2*i], consumableEntries[2*i + 1])

            result = (hiddenArray[0] * weightMatrix1[0][0]) + (hiddenArray[1] * weightMatrix1[1][0]) + (hiddenArray[2] * weightMatrix1[2][0]) + (hiddenArray[3] * weightMatrix1[3][0])
            if result >= 0.5:
                result = 1
                resultColor = ORANGE
            else:
                result = 0
                resultColor = BLUE

            resultText = Text(str(result), font_size = 30).move_to(basicMLP.vertices[7])
            resultText.generate_target()
            resultText.target.shift([1.2, 0, 0]).scale(2.0)

            self.play(basicMLP.vertices[7].animate.set_color(resultColor), run_time = 0.4*speedMult)
            self.play(FadeIn(resultText), MoveToTarget(resultText), run_time = 0.8*(speedMult/2))
            self.wait(0.1*speedMult)

            if i < 3:
                self.play(selectedPoint.animate.set_color(resultColor), FadeOut(pointBox), run_time = 0.3*speedMult2)
            else:
                self.play(selectedPoint.animate.set_color(resultColor), run_time = 0.3*speedMult2)
            self.wait(0.1*speedMult)

            self.play(FadeOut(resultText), run_time = 0.4*(speedMult2))
        
        self.wait(5)


class IntroCredits(Scene):
    def construct(self):
        credits1 = Text("Animation by Kai Gomez", font_size = 45)
        credits2 = Text("Narration by Keyaan Sameer", font_size = 45)
        credits3 = Text("Created for ENGR 2300.001's final project", font_size = 25)
        credits4 = Text("for the Fall 2024 semester at UTD", font_size = 25)

        creditsGroup1 = VGroup(credits1, credits2).arrange(DOWN, buff=0.75).shift([0, 1, 0])
        creditsGroup2 = VGroup(credits3, credits4).arrange(DOWN, buff=0.25).shift([0, -1.25, 0])

        self.play(Write(creditsGroup1), run_time = 1.5)
        self.play(Write(creditsGroup2), run_time = 1.2)
        self.wait(8)

        self.play(Unwrite(creditsGroup2), run_time = 0.75)
        self.play(Unwrite(creditsGroup1), run_time = 0.6)
        self.wait(0.5)

        moreVideo1 = Text("As much of the most interesting information", font_size = 36)
        moreVideo2 = Text("about neural networks, such as backpropagation,", font_size = 36)
        moreVideo3 = Text("has not been covered yet, there may be a second", font_size = 36)
        moreVideo4 = Text("part to this video! Stay tuned.", font_size = 36)

        moreVideoGroup = VGroup(moreVideo1, moreVideo2, moreVideo3, moreVideo4).arrange(DOWN, buff=0.25).shift([0, 0, 0])

        self.play(Write(moreVideoGroup), run_time = 2.5)
        self.wait(5)

        self.play(Unwrite(moreVideoGroup), run_time = 1.25)
        self.wait(0.5)

        thanks = Text("Thank you for watching!", font_size = 54)

        self.play(Write(thanks), run_time = 0.75)
        self.wait(4)

        self.play(FadeOut(thanks), run_time = 2)
        self.wait(5)