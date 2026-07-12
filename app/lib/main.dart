import 'package:flutter/material.dart';

import 'camera_view.dart';

void main() {
  runApp(const RealtimeSegApp());
}

class RealtimeSegApp extends StatelessWidget {
  const RealtimeSegApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Realtime Segmentation',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true),
      home: const Scaffold(
        backgroundColor: Colors.black,
        body: SafeArea(child: CameraSegView()),
      ),
    );
  }
}
