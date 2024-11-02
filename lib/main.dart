import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Mango Disease',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Mango Disease'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final picker = ImagePicker();
  File? _image;
  Interpreter? _interpreter;
  List<String>? _labels;
  String _result = "";

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
      _labels = await loadLabels('assets/labels.txt');
    } catch (e) {
      print("Error loading model: $e");
    }
  }

  Future<List<String>> loadLabels(String assetPath) async {
    final labelsData = await rootBundle.loadString(assetPath);
    return labelsData.split('\n');
  }

  Future<void> classifyImage(File image) async {
    if (_interpreter == null || _labels == null) {
      print("Model or labels not loaded");
      return;
    }

    // Load the image, resize it, and normalize it
    var input = await preprocessImage(image);

    // Prepare output buffer
    var output = List.filled(25200 * 9, 0.0).reshape([1, 25200, 9]);

    // Run inference
    _interpreter!.run(input, output);

    // Process output to get predictions with labels and confidence
    setState(() {
      _result = processOutput(output.cast<List<List<double>>>());
    });
  }

  Future<List<List<List<List<double>>>>> preprocessImage(File imageFile) async {
    // Load image and resize to 640x640
    var image = img.decodeImage(await imageFile.readAsBytes())!;
    var resizedImage = img.copyResize(image, width: 640, height: 640);

    // Initialize input image with shape [1, 640, 640, 3]
    List<List<List<List<double>>>> inputImage = List.generate(
        1,
        (_) => List.generate(
            640, (_) => List.generate(640, (_) => List.filled(3, 0.0))));

    // Normalize pixel values and assign RGB channels
    for (int y = 0; y < 640; y++) {
      for (int x = 0; x < 640; x++) {
        var pixel = resizedImage.getPixel(x, y);

        // Extract RGB channels and normalize to [0, 1]
        inputImage[0][y][x][0] = pixel.r / 255.0; // Red channel
        inputImage[0][y][x][1] = pixel.g / 255.0; // Green channel
        inputImage[0][y][x][2] = pixel.b / 255.0; // Blue channel
      }
    }

    return inputImage;
  }

  String processOutput(List<List<List<double>>> output) {
    String result = "";

    for (var i = 0; i < output[0].length; i++) {
      final confidence = output[0][i][4];
      if (confidence > 0.5) {
        // Adjust the threshold as necessary
        final labelIndex = output[0][i]
            .sublist(5)
            .indexOf(output[0][i].sublist(5).reduce((a, b) => a > b ? a : b));
        final label = _labels![labelIndex];
        result +=
            "$label - Confidence: ${(confidence * 100).toStringAsFixed(2)}%\n";
      }
    }

    return result.isEmpty ? "No disease detected." : result;
  }

  Future<void> getImageFromGallery() async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
      await classifyImage(_image!);
    }
  }

  Future<void> getImageFromCamera() async {
    final pickedFile = await picker.pickImage(source: ImageSource.camera);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
      await classifyImage(_image!);
    }
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Disease Detection"),
      ),
      body: Center(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              _image == null ? Text("No image selected") : Image.file(_image!),
              SizedBox(height: 20),
              Text(
                _result,
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  ElevatedButton(
                    onPressed: getImageFromGallery,
                    child: Text("Pick from Gallery"),
                  ),
                  ElevatedButton(
                    onPressed: getImageFromCamera,
                    child: Text("Capture from Camera"),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
