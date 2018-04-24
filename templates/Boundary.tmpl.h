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
//! \\file {{class_name}}.h
//! \\author pystencils
//======================================================================================================================


#include "core/DataTypes.h"

{% if target is equalto 'cpu' -%}
#include "field/GhostLayerField.h"
{%- elif target is equalto 'gpu' -%}
#include "cuda/GPUField.h"
{%- endif %}
#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/IBlock.h"
#include "blockforest/StructuredBlockForest.h"
#include "field/FlagField.h"

#include <set>
#include <vector>

#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

namespace walberla {
namespace {{namespace}} {


class {{class_name}}
{
public:
    {{StructDeclaration|indent(4)}}

    typedef std::vector<{{StructName}}> IndexVector;

    {{class_name}}( const shared_ptr<StructuredBlockForest> & blocks,
                   {{kernel|generate_constructor_parameters(['indexVector', 'indexVectorSize'])}} )
        : {{ kernel|generate_constructor_initializer_list(['indexVector', 'indexVectorSize']) }}
    {
        auto createIdxVector = []( IBlock * const , StructuredBlockStorage * const ) { return new IndexVector(); };
        indexVectorID = blocks->addStructuredBlockData< IndexVector >( createIdxVector, "IndexField_{{class_name}}");
    };

    void operator() ( IBlock * block );

    template<typename FlagField_T>
    void fillFromFlagField( const shared_ptr<StructuredBlockForest> & blocks, ConstBlockDataID flagFieldID,
                            FlagUID boundaryFlagUID, FlagUID domainFlagUID)
    {
        for( auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt )
            fillFromFlagField<FlagField_T>( &*blockIt, flagFieldID, boundaryFlagUID, domainFlagUID );
    }


    template<typename FlagField_T>
    void fillFromFlagField( IBlock * block, ConstBlockDataID flagFieldID,
                            FlagUID boundaryFlagUID, FlagUID domainFlagUID )
    {
        auto * indexVector = block->getData< IndexVector > ( indexVectorID );
        auto * flagField = block->getData< FlagField_T > ( flagFieldID );

        auto boundaryFlag = flagField->getFlag(boundaryFlagUID);
        auto domainFlag = flagField->getFlag(domainFlagUID);

        indexVector->clear();

        for( auto it = flagField->begin(); it != flagField->end(); ++it )
        {
            if( ! isFlagSet(it, domainFlag) )
                continue;

            {%- for dirIdx, offset in stencil_info %}
            {% if dim == 2 -%}
            if ( isFlagSet( it.neighbor({{offset}}, 0), boundaryFlag ) )
                indexVector->push_back({{StructName}}(it.x(), it.y(), {{dirIdx}} ) );
            {%- elif dim == 3 -%}
            if ( isFlagSet( it.neighbor({{offset}}), boundaryFlag ) )
                indexVector->push_back({{StructName}}(it.x(), it.y(), it.z(), {{dirIdx}} ) );
            {%- endif -%}
            {% endfor %}
        }
    }

private:
    BlockDataID indexVectorID;
    {{kernel|generate_members(['indexVector', 'indexVectorSize'])|indent(4)}}
};



} // namespace {{namespace}}
} // namespace walberla
