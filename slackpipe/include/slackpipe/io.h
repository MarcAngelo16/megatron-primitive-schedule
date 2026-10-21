#pragma once

#include <string>
#include "slackpipe/search_result.h"

namespace slackpipe {
std::string ToJson(const Instance& instance, const ScheduleSolution& schedule);
std::string ToJson(const Instance& instance, const SearchResult& result);
std::string ToCsv(const Instance& instance, const ScheduleSolution& schedule);
std::string ToOrdersText(const Instance& instance, const MachineOrders& orders);
std::string ToSvg(const Instance& instance, const ScheduleSolution& schedule);
void WriteTextFile(const std::string& path, const std::string& contents);
}  // namespace slackpipe
