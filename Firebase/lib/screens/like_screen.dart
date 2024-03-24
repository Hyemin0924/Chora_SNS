import 'package:flutter/material.dart';
import 'package:instagram_clone/exceptions/custom_exception.dart';
import 'package:instagram_clone/models/feed_model.dart';
import 'package:instagram_clone/providers/like/like_provider.dart';
import 'package:instagram_clone/providers/like/like_state.dart';
import 'package:instagram_clone/widgets/error_dialog_widget.dart';
import 'package:instagram_clone/widgets/feed_card_widget.dart';
import 'package:provider/provider.dart';

class LikeScreen extends StatefulWidget {
  const LikeScreen({super.key});

  @override
  State<LikeScreen> createState() => _LikeScreenState();
}

class _LikeScreenState extends State<LikeScreen>
    with AutomaticKeepAliveClientMixin<LikeScreen> {
  final ScrollController _scrollController = ScrollController();
  late final LikeProvider likeProvider;

  @override
  bool get wantKeepAlive => true;

  @override
  void initState() {
    super.initState();
    _scrollController.addListener(scrollListener);
    likeProvider = context.read<LikeProvider>();
    _getLikeList();
  }

  @override
  void dispose() {
    _scrollController.removeListener(scrollListener);
    _scrollController.dispose();
    super.dispose();
  }

  void scrollListener() {
    LikeState likeState = context.read<LikeState>();

    if (likeState.likeStatus == LikeStatus.reFetching) {
      return;
    }

    bool hasNext = likeState.hasNext;

    if (_scrollController.offset >=
            _scrollController.position.maxScrollExtent &&
        hasNext) {
      FeedModel lastFeedModel = likeState.likeList.last;
      context.read<LikeProvider>().getLikeList(
            feedId: lastFeedModel.feedId,
          );
    }
  }

  void _getLikeList() {
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      try {
        await likeProvider.getLikeList();
      } on CustomException catch (e) {
        errorDialogWidget(context, e);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    final likeState = context.watch<LikeState>();
    List<FeedModel> likeList = likeState.likeList;

    if (likeState.likeStatus == LikeStatus.fetching) {
      return Center(
        child: CircularProgressIndicator(),
      );
    }

    if (likeState.likeStatus == LikeStatus.success && likeList.length == 0) {
      return Center(
        child: Text('Feed가 존재하지 않습니다.'),
      );
    }

    return SafeArea(
        child: RefreshIndicator(
      onRefresh: () async {
        _getLikeList();
      },
      child: ListView.builder(
        controller: _scrollController,
        physics: const AlwaysScrollableScrollPhysics(),
        itemCount: likeList.length + 1,
        itemBuilder: (context, index) {
          if (likeList.length == index) {
            return likeState.hasNext
                ? Center(
                    child: Padding(
                      padding: const EdgeInsets.only(bottom: 10),
                      child: CircularProgressIndicator(),
                    ),
                  )
                : Container();
          }

          return FeedCardWidget(
            feedModel: likeList[index],
          );
        },
      ),
    ));
  }
}
