import 'package:flutter/material.dart';
import 'package:instagram_clone/models/feed_model.dart';
import 'package:instagram_clone/providers/profile/profile_state.dart';
import 'package:instagram_clone/widgets/feed_card_widget.dart';
import 'package:provider/provider.dart';

class ProfileFeedScreen extends StatefulWidget {
  final int index;

  const ProfileFeedScreen({
    super.key,
    required this.index,
  });

  @override
  State<ProfileFeedScreen> createState() => _ProfileFeedScreenState();
}

class _ProfileFeedScreenState extends State<ProfileFeedScreen> {
  @override
  Widget build(BuildContext context) {
    List<FeedModel> feedList = context.watch<ProfileState>().feedList;

    return Scaffold(
      body: SafeArea(
        child: FeedCardWidget(
          feedModel: feedList[widget.index],
          isProfile: true,
        ),
      ),
    );
  }
}
