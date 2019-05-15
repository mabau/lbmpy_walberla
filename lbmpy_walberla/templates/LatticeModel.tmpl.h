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
//! \\author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================


#include "core/DataTypes.h"
#include "core/logging/Logging.h"

#include "field/GhostLayerField.h"
#include "field/SwapableCompare.h"
#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/IBlock.h"
#include "stencil/{{stencil_name}}.h"

#include "lbm/lattice_model/EquilibriumDistribution.h"
#include "lbm/field/Density.h"
#include "lbm/field/DensityAndMomentumDensity.h"
#include "lbm/field/DensityAndVelocity.h"
#include "lbm/field/PressureTensor.h"
#include "lbm/field/ShearRate.h"

#include <set>

#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

#ifdef WALBERLA_CXX_COMPILER_IS_GNU
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#ifdef WALBERLA_CXX_COMPILER_IS_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif

{% set lmIgnores = ('pdfs', 'pdfs_tmp') %}


// Forward declarations
namespace walberla{
namespace {{namespace}} {
   class {{class_name}};
}}
namespace walberla {
namespace mpi {
    mpi::SendBuffer & operator<< (mpi::SendBuffer & buf, const ::walberla::{{namespace}}::{{class_name}} & lm);
    mpi::RecvBuffer & operator>> (mpi::RecvBuffer & buf,       ::walberla::{{namespace}}::{{class_name}} & lm);
}}




namespace walberla {
namespace {{namespace}} {


/**
{{class_name}} was generated with lbmpy. Do not edit this file directly. Instead modify {{class_name}}.py.
For details see documentation of lbmpy.

Usage:
    - Create an instance of this lattice model class: the constructor parameters vary depending on the configure
      lattice model. A model with constant force needs a single force vector, while a model with variable forces needs
      a force field. All constructor parameters are ordered alphabetically.
    - Create a PDFField with the lattice model as template argument to store the particle distribution functions.
      Use the PDFField to get and modify macroscopic values.
    - The internal class {{class_name}}::Sweep is a functor to execute one LB time step.
      Stream, collide steps can be executed separately, or together in an optimized stream-pull-collide scheme

*/
class {{class_name}}
{

public:
    typedef stencil::{{stencil_name}} Stencil;
    typedef stencil::{{stencil_name}} CommunicationStencil;
    static const real_t w[{{Q}}];
    static const real_t wInv[{{Q}}];

    static const bool compressible = {% if compressible %}true{% else %}false{% endif %};
    static const int equilibriumAccuracyOrder = {{equilibrium_order}};

    class Sweep
    {
    public:
        Sweep( BlockDataID _pdfsID ) : pdfsID(_pdfsID) {};

        //void stream       ( IBlock * const block, const uint_t numberOfGhostLayersToInclude = uint_t(0) );
        void collide      ( IBlock * const block, const uint_t numberOfGhostLayersToInclude = uint_t(0) );
        void streamCollide( IBlock * const block, const uint_t numberOfGhostLayersToInclude = uint_t(0) );
        void stream       ( IBlock * const block, const uint_t numberOfGhostLayersToInclude = uint_t(0) );

        void operator() ( IBlock * const block, const uint_t numberOfGhostLayersToInclude = uint_t(0) )
        {
            streamCollide( block, numberOfGhostLayersToInclude );
        }

    private:
        {{stream_collide_kernel|generate_members(only_fields=True)|indent(8)}}
    };

    {{class_name}}( {{stream_collide_kernel|generate_constructor_parameters(lmIgnores) }} )
        : {{ stream_collide_kernel|generate_constructor_initializer_list(lmIgnores) }}, currentLevel(0)
    {};

    void configure( IBlock & block, StructuredBlockStorage &)  { configureBlock( &block ); }

    // Parameters:
    {{stream_collide_kernel|generate_members(lmIgnores)|indent(4)}}

private:
    void configureBlock(IBlock * block)
    {
        {{stream_collide_kernel|generate_block_data_to_field_extraction(lmIgnores, no_declarations=True)|indent(8)}}

        {% if refinement_scaling -%}
        const uint_t targetLevel = block->getBlockStorage().getLevel(*block);

        if( targetLevel != currentLevel )
        {
            real_t level_scale_factor(1);
            if( currentLevel < targetLevel )
               level_scale_factor = real_c( uint_t(1) << ( targetLevel - currentLevel ) );
            else // currentLevel > targetLevel
               level_scale_factor = real_t(1) / real_c( uint_t(1) << ( currentLevel - targetLevel ) );

            {% for scalingType, name, expression in refinement_scaling.scaling_info -%}
            {% if scalingType == 'normal' %}
            {{name}} = {{expression}};
            {% elif scalingType in ('field_with_f', 'field_xyz') %}
            auto it = {{name}}->{% if scalingType == 'field_with_f'%} beginWithGhostLayer(){% else %}beginWithGhostLayerXYZ(){% endif %};
            for( ; it != {{name}}->end(); ++it )
            {
                 auto x = it.x();
                 auto y = it.y();
                 auto z = it.z();
                 {% if scalingType == 'field_with_f' -%}
                 auto f = it.f();
                 {% endif -%}
                 *it = {{expression}};
            }
            {% endif -%}
            {% endfor -%}

            currentLevel = targetLevel;
        }
        {% endif -%}
    }

    // Updated by configureBlock:
    {{stream_collide_kernel|generate_block_data_to_field_extraction(lmIgnores, declarations_only=True)|indent(4)}}
    uint_t currentLevel;

    // Backend classes can access private members:
    friend class {{class_name}}::Sweep;
    template<class LM, class Enable> friend class  EquilibriumDistribution;
    template<class LM, class Enable> friend struct Equilibrium;
    template<class LM, class Enable> friend struct internal::AdaptVelocityToForce;
    template<class LM, class Enable> friend struct Density;
    template<class LM>               friend struct DensityAndVelocity;
    template<class LM, class Enable> friend struct DensityAndMomentumDensity;
    template<class LM, class Enable> friend struct MomentumDensity;
    template<class LM, class It, class Enable> friend struct DensityAndVelocityRange;

    friend mpi::SendBuffer & ::walberla::mpi::operator<< (mpi::SendBuffer & , const {{class_name}} & );
    friend mpi::RecvBuffer & ::walberla::mpi::operator>> (mpi::RecvBuffer & ,       {{class_name}} & );

};




//======================================================================================================================
//
//  Implementation of macroscopic value backend
//
//======================================================================================================================



template<>
class EquilibriumDistribution< {{class_name}}, void>
{
public:
   typedef typename {{class_name}}::Stencil Stencil;

   static real_t get( const stencil::Direction direction,
                      const Vector3< real_t > & u = Vector3< real_t >( real_t(0.0) ),
                      real_t rho = real_t(1.0) )
   {
        {% if not compressible %}
        rho -= real_t(1.0);
        {% endif %}
        {{equilibrium_from_direction}}
   }

   static real_t getSymmetricPart( const stencil::Direction direction,
                                   const Vector3<real_t> & u = Vector3< real_t >(real_t(0.0)),
                                   real_t rho = real_t(1.0) )
   {
        {% if not compressible %}
        rho -= real_t(1.0);
        {% endif %}
        {{symmetric_equilibrium_from_direction}}
   }

   static real_t getAsymmetricPart( const stencil::Direction direction,
                                    const Vector3< real_t > & u = Vector3<real_t>( real_t(0.0) ),
                                    real_t rho = real_t(1.0) )
   {
        {% if not compressible %}
        rho -= real_t(1.0);
        {% endif %}
        {{asymmetric_equilibrium_from_direction}}
   }

   static std::vector< real_t > get( const Vector3< real_t > & u = Vector3<real_t>( real_t(0.0) ),
                                     real_t rho = real_t(1.0) )
   {
      {% if not compressible %}
      rho -= real_t(1.0);
      {% endif %}

      std::vector< real_t > equilibrium( Stencil::Size );
      for( auto d = Stencil::begin(); d != Stencil::end(); ++d )
      {
         equilibrium[d.toIdx()] = get(*d, u, rho);
      }
      return equilibrium;
   }
};


namespace internal {

template<>
struct AdaptVelocityToForce<{{class_name}}, void>
{
   template< typename FieldPtrOrIterator >
   static Vector3<real_t> get( FieldPtrOrIterator & it, const {{class_name}} & lm,
                               const Vector3< real_t > & velocity, const real_t rho )
   {
      auto x = it.x();
      auto y = it.y();
      auto z = it.z();
      {% if macroscopic_velocity_shift %}
      return velocity - Vector3<real_t>({{macroscopic_velocity_shift | join(",") }} {% if D == 2 %}, 0.0 {%endif %} );
      {% else %}
      return velocity;
      {% endif %}
   }

   static Vector3<real_t> get( const cell_idx_t x, const cell_idx_t y, const cell_idx_t z, const {{class_name}} & lm,
                               const Vector3< real_t > & velocity, const real_t rho )
   {
      {% if macroscopic_velocity_shift %}

      return velocity - Vector3<real_t>({{macroscopic_velocity_shift | join(",") }} {% if D == 2 %}, 0.0 {%endif %} );
      {% else %}
      return velocity;
      {% endif %}
   }
};
} // namespace internal



template<>
struct Equilibrium< {{class_name}}, void >
{

   template< typename FieldPtrOrIterator >
   static void set( FieldPtrOrIterator & it,
                    const Vector3< real_t > & u = Vector3< real_t >( real_t(0.0) ), real_t rho = real_t(1.0) )
   {
        {%if not compressible %}
        rho -= real_t(1.0);
        {%endif %}

       {% for eqTerm in equilibrium -%}
       it[{{loop.index0 }}] = {{eqTerm}};
       {% endfor -%}
   }

   template< typename PdfField_T >
   static void set( PdfField_T & pdf, const cell_idx_t x, const cell_idx_t y, const cell_idx_t z,
                    const Vector3< real_t > & u = Vector3< real_t >( real_t(0.0) ), real_t rho = real_t(1.0) )
   {
      {%if not compressible %}
      rho -= real_t(1.0);
      {%endif %}

      real_t & xyz0 = pdf(x,y,z,0);
      {% for eqTerm in equilibrium -%}
      pdf.getF( &xyz0, {{loop.index0 }})= {{eqTerm}};
      {% endfor -%}
   }
};


template<>
struct Density<{{class_name}}, void>
{
   template< typename FieldPtrOrIterator >
   static inline real_t get( const {{class_name}} & , const FieldPtrOrIterator & it )
   {
        {% for i in range(Q) -%}
            const real_t f_{{i}} = it[{{i}}];
        {% endfor -%}
        {{density_getters | indent(8)}}
        return rho;
   }

   template< typename PdfField_T >
   static inline real_t get( const {{class_name}} & ,
                             const PdfField_T & pdf, const cell_idx_t x, const cell_idx_t y, const cell_idx_t z )
   {
        const real_t & xyz0 = pdf(x,y,z,0);
        {% for i in range(Q) -%}
            const real_t f_{{i}} = pdf.getF( &xyz0, {{i}});
        {% endfor -%}
        {{density_getters | indent(8)}}
        return rho;
   }
};


template<>
struct DensityAndVelocity<{{class_name}}>
{
    template< typename FieldPtrOrIterator >
    static void set( FieldPtrOrIterator & it, const {{class_name}} & lm,
                     const Vector3< real_t > & u = Vector3< real_t >( real_t(0.0) ), const real_t rho_in = real_t(1.0) )
    {
        auto x = it.x();
        auto y = it.y();
        auto z = it.z();

        {{density_velocity_setter_macroscopic_values | indent(8)}}
        {% if D == 2 -%}
        const real_t u_2(0.0);
        {% endif %}

        Equilibrium<{{class_name}}>::set(it, Vector3<real_t>(u_0, u_1, u_2), rho{%if not compressible %} + real_t(1) {%endif%});
    }

    template< typename PdfField_T >
    static void set( PdfField_T & pdf, const cell_idx_t x, const cell_idx_t y, const cell_idx_t z, const {{class_name}} & lm,
                     const Vector3< real_t > & u = Vector3< real_t >( real_t(0.0) ), const real_t rho_in = real_t(1.0) )
    {
        {{density_velocity_setter_macroscopic_values | indent(8)}}
        {% if D == 2 -%}
        const real_t u_2(0.0);
        {% endif %}

        Equilibrium<{{class_name}}>::set(pdf, x, y, z, Vector3<real_t>(u_0, u_1, u_2), rho {%if not compressible %} + real_t(1) {%endif%});
    }
};


template<typename FieldIteratorXYZ >
struct DensityAndVelocityRange<{{class_name}}, FieldIteratorXYZ>
{

   static void set( FieldIteratorXYZ & begin, const FieldIteratorXYZ & end, const {{class_name}} & lm,
                    const Vector3< real_t > & u = Vector3< real_t >( real_t(0.0) ), const real_t rho_in = real_t(1.0) )
   {
        for( auto cellIt = begin; cellIt != end; ++cellIt )
        {
            const auto x = cellIt.x();
            const auto y = cellIt.y();
            const auto z = cellIt.z();
            {{density_velocity_setter_macroscopic_values | indent(12)}}
            {% if D == 2 -%}
            const real_t u_2(0.0);
            {% endif %}

            Equilibrium<{{class_name}}>::set(cellIt, Vector3<real_t>(u_0, u_1, u_2), rho{%if not compressible %} + real_t(1) {%endif%});
        }
   }
};



template<>
struct DensityAndMomentumDensity<{{class_name}}>
{
   template< typename FieldPtrOrIterator >
   static real_t get( Vector3< real_t > & momentumDensity, const {{class_name}} & lm,
                      const FieldPtrOrIterator & it )
   {
        const auto x = it.x();
        const auto y = it.y();
        const auto z = it.z();

        {% for i in range(Q) -%}
            const real_t f_{{i}} = it[{{i}}];
        {% endfor -%}

        {{momentum_density_getter | indent(8) }}
        {% for i in range(D) -%}
            momentumDensity[{{i}}] = md_{{i}};
        {% endfor %}
        return rho;
   }

   template< typename PdfField_T >
   static real_t get( Vector3< real_t > & momentumDensity, const {{class_name}} & lm, const PdfField_T & pdf,
                      const cell_idx_t x, const cell_idx_t y, const cell_idx_t z )
   {
        const real_t & xyz0 = pdf(x,y,z,0);
        {% for i in range(Q) -%}
            const real_t f_{{i}} = pdf.getF( &xyz0, {{i}});
        {% endfor -%}

        {{momentum_density_getter | indent(8) }}
        {% for i in range(D) -%}
            momentumDensity[{{i}}] = md_{{i}};
        {% endfor %}
       return rho;
   }
};


template<>
struct MomentumDensity< {{class_name}}>
{
   template< typename FieldPtrOrIterator >
   static void get( Vector3< real_t > & momentumDensity, const {{class_name}} & lm, const FieldPtrOrIterator & it )
   {
        const auto x = it.x();
        const auto y = it.y();
        const auto z = it.z();

        {% for i in range(Q) -%}
            const real_t f_{{i}} = it[{{i}}];
        {% endfor -%}

        {{momentum_density_getter | indent(8) }}
        {% for i in range(D) -%}
            momentumDensity[{{i}}] = md_{{i}};
        {% endfor %}
   }

   template< typename PdfField_T >
   static void get( Vector3< real_t > & momentumDensity, const {{class_name}} & lm, const PdfField_T & pdf,
                    const cell_idx_t x, const cell_idx_t y, const cell_idx_t z )
   {
        const real_t & xyz0 = pdf(x,y,z,0);
        {% for i in range(Q) -%}
            const real_t f_{{i}} = pdf.getF( &xyz0, {{i}});
        {% endfor -%}

        {{momentum_density_getter | indent(8) }}
        {% for i in range(D) -%}
            momentumDensity[{{i}}] = md_{{i}};
        {% endfor %}
   }
};


template<>
struct PressureTensor<{{class_name}}>
{
   template< typename FieldPtrOrIterator >
   static void get( Matrix3< real_t > & /* pressureTensor */, const {{class_name}} & /* latticeModel */, const FieldPtrOrIterator & /* it */ )
   {
       WALBERLA_ABORT("Not implemented");
   }

   template< typename PdfField_T >
   static void get( Matrix3< real_t > & /* pressureTensor */, const {{class_name}} & /* latticeModel */, const PdfField_T & /* pdf */,
                    const cell_idx_t /* x */, const cell_idx_t /* y */, const cell_idx_t /* z */ )
   {
       WALBERLA_ABORT("Not implemented");
   }
};


template<>
struct ShearRate<{{class_name}}>
{
   template< typename FieldPtrOrIterator >
   static inline real_t get( const {{class_name}} & /* latticeModel */, const FieldPtrOrIterator & /* it */,
                             const Vector3< real_t > & /* velocity */, const real_t /* rho */)
   {
       WALBERLA_ABORT("Not implemented");
       return real_t(0.0);
   }

   template< typename PdfField_T >
   static inline real_t get( const {{class_name}} & latticeModel,
                             const PdfField_T & /* pdf */, const cell_idx_t /* x */, const cell_idx_t /* y */, const cell_idx_t /* z */,
                             const Vector3< real_t > & /* velocity */, const real_t /* rho */ )
   {
       WALBERLA_ABORT("Not implemented");
       return real_t(0.0);
   }

   static inline real_t get( const std::vector< real_t > & /* nonEquilibrium */, const real_t /* relaxationParam */,
                             const real_t /* rho */ = real_t(1) )
   {
       WALBERLA_ABORT("Not implemented");
       return real_t(0.0);
   }
};


} // namespace {{namespace}}
} // namespace walberla



#ifdef WALBERLA_CXX_COMPILER_IS_GNU
#pragma GCC diagnostic pop
#endif

#ifdef WALBERLA_CXX_COMPILER_IS_CLANG
#pragma clang diagnostic pop
#endif
