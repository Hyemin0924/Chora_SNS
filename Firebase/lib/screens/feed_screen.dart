import 'package:flutter/material.dart';
import 'package:instagram_clone/exceptions/custom_exception.dart';
import 'package:instagram_clone/models/feed_model.dart';
import 'package:instagram_clone/providers/feed/feed_provider.dart';
import 'package:instagram_clone/providers/feed/feed_state.dart';
import 'package:instagram_clone/widgets/error_dialog_widget.dart';
import 'package:instagram_clone/widgets/feed_card_widget.dart';
import 'package:provider/provider.dart';

class FeedScreen extends StatefulWidget {
  const FeedScreen({super.key});

  @override
  State<FeedScreen> createState() => _FeedScreenState();
}

class _FeedScreenState extends State<FeedScreen>
    with AutomaticKeepAliveClientMixin<FeedScreen> {
  final ScrollController _scrollController = ScrollController();
  late final FeedProvider feedProvider;

  @override
  bool get wantKeepAlive => true;

  @override
  void initState() {
    super.initState();
    feedProvider = context.read<FeedProvider>();
    _scrollController.addListener(scrollListener);
    _getFeedList();
  }

  @override
  void dispose() {
    _scrollController.removeListener(scrollListener);
    _scrollController.dispose();
    super.dispose();
  }

  void scrollListener() {
    FeedState feedState = context.read<FeedState>();

    if (feedState.feedStatus == FeedStatus.reFetching) {
      return;
    }

    bool hasNext = feedState.hasNext;

    if (_scrollController.offset >=
            _scrollController.position.maxScrollExtent &&
        hasNext) {
      FeedModel lastFeedModel = feedState.feedList.last;
      context.read<FeedProvider>().getFeedList(
            feedId: lastFeedModel.feedId,
          );
    }
  }

  void _getFeedList() {
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      try {
        await feedProvider.getFeedList();
      } on CustomException catch (e) {
        errorDialogWidget(context, e);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    FeedState feedState = context.watch<FeedState>();
    List<FeedModel> feedList = feedState.feedList;

    if (feedState.feedStatus == FeedStatus.fetching) {
      return Center(
        child: CircularProgressIndicator(),
      );
    }

    if (feedState.feedStatus == FeedStatus.success && feedList.length == 0) {
      return Center(
        child: Text('Feed가 존재하지 않습니다.'),
      );
    }

    return SafeArea(
      child: RefreshIndicator(
        onRefresh: () async {
          _getFeedList();
        },
        child: ListView.builder(
          controller: _scrollController,
          physics: const AlwaysScrollableScrollPhysics(),
          itemCount: feedList.length + 1,
          itemBuilder: (context, index) {
            if (feedList.length == index) {
              return feedState.hasNext
                  ? Center(
                      child: Padding(
                        padding: const EdgeInsets.only(bottom: 10),
                        child: CircularProgressIndicator(),
                      ),
                    )
                  : Container();
            }

            return FeedCardWidget(feedModel: feedList[index]);
          },
        ),
      ),
    );
  }
}
