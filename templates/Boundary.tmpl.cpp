//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \\file {{class_name}}.cpp
//! \\ingroup lbm
//! \\author lbmpy
//======================================================================================================================

#include <cmath>

#include "core/DataTypes.h"
#include "core/Macros.h"
#include "{{class_name}}.h"


{% if target is equalto 'cpu' -%}
#define FUNC_PREFIX
{%- elif target is equalto 'gpu' -%}
#define FUNC_PREFIX __global__
{%- endif %}

using namespace std;

namespace walberla {
namespace {{namespace}} {

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

{{kernel|generate_definition}}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif


void {{class_name}}::operator() ( IBlock * block )
{
    auto  * indexVector = block->getData<IndexVector>(indexVectorID);

    int64_t indexVectorSize = int64_c( indexVector->size() );
    uint8_t * fd_indexVector = reinterpret_cast<uint8_t*>(&(*indexVector)[0]);
    {{kernel|generate_block_data_to_field_extraction(['indexVector', 'indexVectorSize'])|indent(4)}}
    {{kernel|generate_call|indent(4)}}
}



} // namespace {{namespace}}
} // namespace walberla
